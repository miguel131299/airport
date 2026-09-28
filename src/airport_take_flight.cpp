#include "airport_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"

// Arrow includes.
#include <arrow/flight/client.h>
#include <arrow/flight/types.h>
#include <arrow/buffer.h>
#include <arrow/util/uri.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/filesystem/localfs.h>

#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "airport_flight_exception.hpp"
#include "airport_flight_statistics.hpp"
#include "airport_flight_stream.hpp"
#include "airport_json_common.hpp"
#include "airport_json_serializer.hpp"
#include "airport_macros.hpp"
#include "airport_request_headers.hpp"
#include "airport_schema_utils.hpp"
#include "airport_take_flight.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "msgpack.hpp"
#include "storage/airport_catalog.hpp"
#include "storage/airport_table_entry.hpp"
#include <openssl/bio.h>
#include <mutex>
#include <random>
#include <unordered_map>
#include <openssl/evp.h>
#include "airport_rpc.hpp"
#include "airport_logging.hpp"

namespace duckdb
{
  // Create a FlightDescriptor from a DuckDB value which can be one of a few different
  // types.
  static flight::FlightDescriptor flight_descriptor_from_value(duckdb::Value &flight_descriptor)
  {
    switch (flight_descriptor.type().id())
    {
    case LogicalTypeId::BLOB:
    case LogicalTypeId::VARCHAR:
      return flight::FlightDescriptor::Command(flight_descriptor.ToString());
    case LogicalTypeId::LIST:
    {
      auto &list_values = ListValue::GetChildren(flight_descriptor);
      vector<string> components;
      for (idx_t i = 0; i < list_values.size(); i++)
      {
        auto &child = list_values[i];
        if (child.type().id() != LogicalTypeId::VARCHAR)
        {
          throw InvalidInputException("airport_take_flight: when specifying a path all list components must be a varchar");
        }
        components.emplace_back(child.ToString());
      }
      return flight::FlightDescriptor::Path(components);
    }
    case LogicalTypeId::ARRAY:
    {
      auto &array_values = ArrayValue::GetChildren(flight_descriptor);
      vector<string> components;
      for (idx_t i = 0; i < array_values.size(); i++)
      {
        auto &child = array_values[i];
        if (child.type().id() != LogicalTypeId::VARCHAR)
        {
          throw InvalidInputException("airport_take_flight: when specifying a path all list components must be a varchar");
        }
        components.emplace_back(child.ToString());
      }
      return flight::FlightDescriptor::Path(components);
    }
    // FIXME: deal with the union type returned by Arrow list flights.
    default:
      throw InvalidInputException("airport_take_flight: unknown descriptor type passed");
    }
  }

  // The FBL server's narrow-decimal rule (x-fbl-narrow-decimals): top-level
  // decimal128 fields of precision <= 9 become decimal32, <= 18 decimal64.
  static std::shared_ptr<arrow::Schema> AirportNarrowDecimalSchema(const std::shared_ptr<arrow::Schema> &schema)
  {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(schema->num_fields());
    bool changed = false;
    for (const auto &field : schema->fields())
    {
      if (field->type()->id() == arrow::Type::DECIMAL128)
      {
        const auto &decimal = static_cast<const arrow::Decimal128Type &>(*field->type());
        if (decimal.precision() <= 9)
        {
          fields.push_back(field->WithType(arrow::decimal32(decimal.precision(), decimal.scale())));
          changed = true;
          continue;
        }
        if (decimal.precision() <= 18)
        {
          fields.push_back(field->WithType(arrow::decimal64(decimal.precision(), decimal.scale())));
          changed = true;
          continue;
        }
      }
      fields.push_back(field);
    }
    return changed ? arrow::schema(std::move(fields), schema->metadata()) : schema;
  }

  unique_ptr<FunctionData>
  AirportTakeFlightBindWithFlightDescriptor(
      const AirportTakeFlightParameters &take_flight_params,
      const flight::FlightDescriptor &descriptor,
      ClientContext &context,
      const TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names,
      // So rather than the cached_flight_info_ptr here we can just have the cached schema.
      std::shared_ptr<arrow::Schema> schema,
      const int64_t estimated_records_hint,
      const std::optional<AirportTableFunctionFlightInfoParameters> &table_function_parameters,
      const AirportTableEntry *table_entry)
  {
    // Create a UID for tracing.
    const auto trace_uuid = airport_trace_id();

    auto &server_location = take_flight_params.server_location();

    AirportLocationDescriptor location_descriptor(server_location, descriptor);

    // The main thing that needs to be answered in this function
    // are the names and return types and establishing the bind data.
    //
    // If the cached_flight_info_ptr is not null we should use the schema
    // from that flight info otherwise it should be requested.

    arrow::flight::FlightCallOptions call_options;
    airport_add_normal_headers(call_options, take_flight_params, trace_uuid,
                               descriptor);

    int64_t estimated_records = estimated_records_hint;

    // If we are applying time travel, the schema that we have is the latest schema
    // but back in time the schema may have been different.
    //
    // We should just request the schema from the server if we are using time travel.
    // Additionally the number of estimated records may also be different.

    if (!take_flight_params.at_unit().empty() && !take_flight_params.at_value().empty())
    {
      schema = nullptr;
    }

    // Get the information about the flight, this will allow the
    // endpoint information to be returned.

    if (schema == nullptr)
    {
      std::unique_ptr<arrow::flight::FlightInfo> retrieved_flight_info;
      auto flight_client = AirportAPI::FlightClientForLocation(server_location);

      if (table_function_parameters != std::nullopt)
      {
        // Rather than calling GetFlightInfo we will call DoAction and get
        // get the flight info that way, since it allows us to serialize
        // all of the data we need to send instead of just the flight name.

        AirportTableFunctionFlightInfoParameters augmented_parameters(*table_function_parameters);

        augmented_parameters.at_unit = take_flight_params.at_unit();
        augmented_parameters.at_value = take_flight_params.at_value();
        AIRPORT_MSGPACK_ACTION_SINGLE_PARAMETER(action, "table_function_flight_info", augmented_parameters);

        auto serialized_flight_info_buffer = AirportCallAction(flight_client, call_options, action, server_location);

        std::string_view serialized_flight_info(reinterpret_cast<const char *>(serialized_flight_info_buffer->body->data()), serialized_flight_info_buffer->body->size());

        // Now deserialize that flight info so we can use it.
        AIRPORT_ASSIGN_OR_RAISE_CONTAINER(retrieved_flight_info, arrow::flight::FlightInfo::Deserialize(serialized_flight_info), &location_descriptor, "deserialize flight info");
      }
      else if (table_entry != nullptr)
      {
        // We have a table entry, which means this isn't an adhoc call to airport_take_flight, so we can call
        // the flight_info action rather than GetFlightInfo which allows additional parameters to be passed.
        AirportFlightInfoParameters get_flight_info_params;

        AIRPORT_ASSIGN_OR_RAISE_CONTAINER(
            get_flight_info_params.descriptor,
            descriptor.SerializeToString(),
            &location_descriptor,
            "airport_take_flight: serialize flight descriptor");

        get_flight_info_params.at_unit = take_flight_params.at_unit();
        get_flight_info_params.at_value = take_flight_params.at_value();

        AIRPORT_MSGPACK_ACTION_SINGLE_PARAMETER(action, "flight_info", get_flight_info_params);

        auto serialized_flight_info_buffer = AirportCallAction(flight_client, call_options, action, server_location);

        std::string_view serialized_flight_info(reinterpret_cast<const char *>(serialized_flight_info_buffer->body->data()), serialized_flight_info_buffer->body->size());

        // Now deserialize that flight info so we can use it.
        AIRPORT_ASSIGN_OR_RAISE_CONTAINER(retrieved_flight_info, arrow::flight::FlightInfo::Deserialize(serialized_flight_info), &location_descriptor, "deserialize flight info");
      }
      else
      {
        AIRPORT_ASSIGN_OR_RAISE_CONTAINER(retrieved_flight_info,
                                          flight_client->GetFlightInfo(call_options, descriptor),
                                          &location_descriptor,
                                          "GetFlightInfo");
      }

      // Assert that the descriptor is the same as the one that was passed in.
      if (descriptor != retrieved_flight_info->descriptor())
      {
        throw InvalidInputException("airport_take_flight: descriptor returned from server does not match the descriptor that was passed in to GetFlightInfo, check with Flight server implementation.");
      }

      estimated_records = retrieved_flight_info->total_records();

      arrow::ipc::DictionaryMemo dictionary_memo;
      AIRPORT_ASSIGN_OR_RAISE_CONTAINER(schema,
                                        retrieved_flight_info->GetSchema(&dictionary_memo),
                                        &location_descriptor,
                                        "GetSchema");
    }

    auto ret = make_uniq<AirportTakeFlightBindData>(
        (stream_factory_produce_t)&AirportCreateStream,
        trace_uuid,
        estimated_records,
        take_flight_params,
        table_function_parameters,
        schema,
        descriptor,
        table_entry,
        nullptr);

    if (table_entry != nullptr)
    {
      auto &catalog = table_entry->GetCatalog().Cast<AirportCatalog>();
      if (catalog.attach_parameters()->fbl_pushdown_enabled())
      {
        const auto &capabilities = table_entry->table_data->fbl_capabilities();
        ret->fbl_projection_pushdown = capabilities.projection_pushdown;
        ret->fbl_exact_filter_pushdown = capabilities.exact_filter_pushdown;
        ret->fbl_hint_filters = capabilities.hint_filters &&
                                catalog.attach_parameters()->fbl_hint_filters_enabled();
        ret->fbl_partial_aggregates = capabilities.partial_aggregates;
      }
    }

    AirportExamineSchema(context,
                         ret->schema_root,
                         &ret->arrow_table,
                         &return_types,
                         &names,
                         nullptr,
                         &ret->rowid_column_index,
                         true);

    // Store the return types and names so they can be
    // validated by parquet_scans or other scans used in endpoints.
    ret->set_types_and_names(return_types, names);
    // AirportArrowScanInitGlobal reads all_types for the scanned columns when
    // projection_ids is populated, and the static-filter expression types its
    // column references from it. The aggregate rewrite replaces it together
    // with the partial schema.
    ret->all_types = return_types;

    return ret;
  }

  static unique_ptr<FunctionData> take_flight_bind(
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names)
  {
    auto server_location = input.inputs[0].ToString();
    AirportTakeFlightParameters params(server_location, context, input);
    auto descriptor = flight_descriptor_from_value(input.inputs[1]);

    return AirportTakeFlightBindWithFlightDescriptor(
        params,
        descriptor,
        context,
        input, return_types, names, nullptr, -1, std::nullopt, nullptr);
  }

  static unique_ptr<FunctionData> take_flight_bind_with_pointer(
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names)
  {
    if (input.inputs[0].IsNull())
    {
      throw BinderException("airport: take_flight_with_pointer, pointers to AirportTable cannot be null");
    }

    if (input.inputs[1].IsNull())
    {
      throw BinderException("airport: take_flight_with_pointer, pointers to AirportTable cannot be null");
    }

    const auto info = reinterpret_cast<const duckdb::AirportAPITable *>(input.inputs[0].GetPointer());
    const auto table_entry = reinterpret_cast<const AirportTableEntry *>(input.inputs[1].GetPointer());

    AirportTakeFlightParameters params(info->server_location(), context, input);

    // Set the catalog name from the table entry's catalog.
    auto &airport_catalog = table_entry->GetCatalog().Cast<AirportCatalog>();
    params.set_catalog_name(airport_catalog.internal_name());
    if (airport_catalog.attach_parameters()->max_endpoints() > 1)
    {
      params.add_header("x-fbl-max-endpoints",
                        std::to_string(airport_catalog.attach_parameters()->max_endpoints()));
    }
    // Sent on every scan RPC; a server that does not know the header ignores it
    // and keeps writing uncompressed IPC, which Arrow reads either way.
    if (!airport_catalog.attach_parameters()->ipc_compression().empty())
    {
      params.add_header("x-fbl-ipc-compression",
                        airport_catalog.attach_parameters()->ipc_compression());
    }
    // Narrow decimals: a capable server ships decimals of precision <= 9 / 18
    // as decimal32 / decimal64 on every DoGet that carries this header, so
    // the schema this scan binds (the catalog's cached one below) must use
    // the same widths. The DuckDB types stay DECIMAL(p, s) either way; only
    // the Arrow import changes, to a zero-copy one.
    auto bind_schema = info->schema();
    if (airport_catalog.attach_parameters()->fbl_pushdown_enabled() &&
        airport_catalog.attach_parameters()->fbl_narrow_decimals_enabled() &&
        table_entry->table_data->fbl_capabilities().narrow_decimals &&
        bind_schema != nullptr)
    {
      params.add_header("x-fbl-narrow-decimals", "1");
      bind_schema = AirportNarrowDecimalSchema(bind_schema);
    }

    // The transaction identifier is passed as the 2nd argument.
    if (!input.inputs[2].IsNull())
    {
      auto id = input.inputs[2].ToString();
      if (!id.empty())
      {
        params.add_header("airport-transaction-id", id);
      }
    }

    return AirportTakeFlightBindWithFlightDescriptor(
        params,
        info->descriptor(),
        context,
        input,
        return_types,
        names,
        bind_schema,
        info->total_records(),
        std::nullopt,
        table_entry);
  }

  static bool
  AirportLocalStateProcessEndpoint(ClientContext &context,
                                   const TableFunctionInitInput &input,
                                   const AirportTakeFlightBindData &bind_data,
                                   AirportArrowScanGlobalState &global_state,
                                   AirportArrowScanLocalState &local_state,
                                   const flight::FlightEndpoint endpoint);

  static bool AirportArrowScanParallelStateNext(AirportArrowScanLocalState &state,
                                                AirportArrowScanGlobalState &global_state,
                                                const AirportTakeFlightBindData &bind_data,
                                                ClientContext &context)
  {
    if (state.done)
    {
      return false;
    }
    state.Reset();
    state.batch_index++; //= ++parallel_state.batch_index;

    bool finished_chunk = false;
    auto &reader = state.reader();
    if (std::holds_alternative<std::shared_ptr<AirportLocalScanData>>(reader))
    {
      auto &scan_data = std::get<std::shared_ptr<AirportLocalScanData>>(reader);
      if (scan_data->finished_chunk)
      {
        finished_chunk = true;
      }
    }
    else
    {
      auto current_chunk = state.stream()->GetNextChunk();
      while (current_chunk->arrow_array.length == 0 && current_chunk->arrow_array.release)
      {
        current_chunk = state.stream()->GetNextChunk();
      }
      state.chunk = std::move(current_chunk);

      finished_chunk = !state.chunk->arrow_array.release;
    }

    //! have we run out of chunks? we are done
    if (finished_chunk)
    {
      auto &endpoint_opt = global_state.GetNextEndpoint();
      if (endpoint_opt)
      {
        if (AirportLocalStateProcessEndpoint(context,
                                             state.input(),
                                             bind_data,
                                             global_state,
                                             state,
                                             *endpoint_opt))
        {
          return true;
        }
      }
      state.done = true;
      return false;
    }
    return true;
  }

  static void AirportDataFromLocalScanFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output)
  {
    auto &state = data_p.local_state->Cast<AirportArrowScanLocalState>();

    auto &reader = state.reader();

    D_ASSERT(std::holds_alternative<std::shared_ptr<AirportLocalScanData>>(reader));

    auto &scan_data = std::get<std::shared_ptr<AirportLocalScanData>>(reader);

    TableFunctionInput function_input(scan_data->bind_data.get(),
                                      scan_data->local_state.get(),
                                      scan_data->global_state.get());
    scan_data->table_function.function(context, function_input, output);

    for (auto &idx : scan_data->not_mapped_column_indexes)
    {
      auto &vec = output.data[idx];
      vec.SetVectorType(VectorType::FLAT_VECTOR);
      FlatVector::Validity(vec).SetAllInvalid(output.size());
    }

    auto count = output.size();
    scan_data->finished_chunk = count == 0;
    output.Verify();
  }

  static void AirportDataFromStream(ClientContext &context, TableFunctionInput &data_p, DataChunk &output)
  {
    auto &state = data_p.local_state->Cast<AirportArrowScanLocalState>();
    auto &global_state = data_p.global_state->Cast<AirportArrowScanGlobalState>();
    auto &airport_bind_data = data_p.bind_data->CastNoConst<AirportTakeFlightBindData>();

    D_ASSERT(!std::holds_alternative<std::shared_ptr<AirportLocalScanData>>(state.reader()));

    const auto &array_length = (idx_t)state.chunk->arrow_array.length;

    // AirportTakeFlight treats an empty output as "this record batch is done",
    // so a slice whose rows the static filters all reject must not end the
    // batch early: keep converting slices until one survives or the batch is
    // exhausted. chunk_offset advances by rows consumed, not rows emitted.
    while (true)
    {
      const auto output_size =
          MinValue<int64_t>(STANDARD_VECTOR_SIZE,
                            array_length - state.chunk_offset);

      if (global_state.CanRemoveFilterColumns())
      {
        // So state.all_columns is a smaller DataChunk, that
        // should just contain the number of columns taht are in the all
        // columns vector.
        state.all_columns.Reset();
        state.all_columns.SetCardinality(output_size);
        if (output_size > 0)
        {
          ArrowTableFunction::ArrowToDuckDB(state,
                                            state.flight_output_columns(airport_bind_data.arrow_table),
                                            state.all_columns,
                                            false,
                                            airport_bind_data.rowid_column_index);
          if (state.static_filter_executor)
          {
            const auto kept = state.static_filter_executor->SelectExpression(state.all_columns,
                                                                             state.static_filter_sel);
            if (kept < (idx_t)output_size)
              state.all_columns.Slice(state.static_filter_sel, kept);
          }
        }
        output.ReferenceColumns(state.all_columns, global_state.projection_ids());
      }
      else
      {
        output.Reset();
        output.SetCardinality(output_size);
        if (output_size > 0)
        {
          ArrowTableFunction::ArrowToDuckDB(state,
                                            state.flight_output_columns(airport_bind_data.arrow_table),
                                            output,
                                            false,
                                            airport_bind_data.rowid_column_index);
          if (state.static_filter_executor)
          {
            const auto kept = state.static_filter_executor->SelectExpression(output, state.static_filter_sel);
            if (kept < (idx_t)output_size)
              output.Slice(state.static_filter_sel, kept);
          }
        }
      }

      state.chunk_offset += output_size;
      if (output.size() != 0 || output_size <= 0 || state.chunk_offset >= array_length)
        break;
    }
    output.Verify();
  }

  void AirportTakeFlight(ClientContext &context, TableFunctionInput &data_p, DataChunk &output)
  {
    // If the local state is null, it means there were no endpoints to scan,
    // so just return the empty output.
    if (data_p.local_state == nullptr)
    {
      DUCKDB_LOG(context, AirportLogType, "Take Flight", {{"status", "No endpoints to scan"}});
      output.SetCardinality(0);
      return;
    }

    D_ASSERT(data_p.global_state);
    D_ASSERT(data_p.bind_data);
    auto &state = data_p.local_state->Cast<AirportArrowScanLocalState>();
    auto &global_state = data_p.global_state->Cast<AirportArrowScanGlobalState>();
    auto &airport_bind_data = data_p.bind_data->CastNoConst<AirportTakeFlightBindData>();

    while (true)
    {
      auto &reader = state.reader();
      const auto has_local_scan = std::holds_alternative<std::shared_ptr<AirportLocalScanData>>(reader);
      if (has_local_scan)
      {
        AirportDataFromLocalScanFunction(context, data_p, output);
      }
      else
      {
        AirportDataFromStream(context, data_p, output);
      }

      if (output.size() != 0)
      {
        break;
      }

      if (!AirportArrowScanParallelStateNext(state,
                                             global_state,
                                             airport_bind_data,
                                             context))
      {
        break;
      }
    }
  }

  InsertionOrderPreservingMap<string> AirportTakeFlightToString(TableFunctionToStringInput &input)
  {
    InsertionOrderPreservingMap<string> result;
    auto &bind_data = input.bind_data->Cast<AirportTakeFlightBindData>();
    auto table_entry = bind_data.table_entry();
    if (table_entry)
    {
      result["Table"] = table_entry->table_data->name();
    }
    result["Server"] = bind_data.server_location();
    result["Descriptor"] = bind_data.descriptor().ToString();
    return result;
  }

  unique_ptr<NodeStatistics> AirportTakeFlightCardinality(ClientContext &context, const FunctionData *data)
  {
    // To estimate the cardinality of the flight, we can peek at the flight information
    // that was retrieved during the bind function.
    //
    // This estimate does not take into account any filters that may have been applied
    //
    auto &bind_data = data->Cast<AirportTakeFlightBindData>();
    auto flight_estimated_records = bind_data.estimated_records();

    if (flight_estimated_records != -1)
    {
      return make_uniq<NodeStatistics>(flight_estimated_records);
    }
    // If we don't have an estimated number of records, just use an assumption.
    return make_uniq<NodeStatistics>();
  }

  static bool AirportIsColumnAndConstant(const Expression &left, const Expression &right)
  {
    return left.expression_class == ExpressionClass::BOUND_COLUMN_REF &&
           right.expression_class == ExpressionClass::BOUND_CONSTANT;
  }

  bool AirportIsExactFblFilter(const Expression &expression)
  {
    switch (expression.expression_class)
    {
    case ExpressionClass::BOUND_COMPARISON:
    {
      const auto &comparison = expression.Cast<BoundComparisonExpression>();
      const auto supported_operator =
          comparison.type == ExpressionType::COMPARE_EQUAL ||
          comparison.type == ExpressionType::COMPARE_NOTEQUAL ||
          comparison.type == ExpressionType::COMPARE_LESSTHAN ||
          comparison.type == ExpressionType::COMPARE_LESSTHANOREQUALTO ||
          comparison.type == ExpressionType::COMPARE_GREATERTHAN ||
          comparison.type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
      return supported_operator &&
             (AirportIsColumnAndConstant(*comparison.left, *comparison.right) ||
              AirportIsColumnAndConstant(*comparison.right, *comparison.left));
    }
    case ExpressionClass::BOUND_BETWEEN:
    {
      const auto &between = expression.Cast<BoundBetweenExpression>();
      return between.input->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
             between.lower->expression_class == ExpressionClass::BOUND_CONSTANT &&
             between.upper->expression_class == ExpressionClass::BOUND_CONSTANT;
    }
    case ExpressionClass::BOUND_CONJUNCTION:
    {
      const auto &conjunction = expression.Cast<BoundConjunctionExpression>();
      if (conjunction.type != ExpressionType::CONJUNCTION_AND || conjunction.children.empty())
        return false;
      return std::all_of(conjunction.children.begin(), conjunction.children.end(),
                         [](const unique_ptr<Expression> &child)
                         { return AirportIsExactFblFilter(*child); });
    }
    default:
      return false;
    }
  }

  void AirportTakeFlightComplexFilterPushdown(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                                              vector<unique_ptr<Expression>> &filters)
  {
    auto allocator = AirportJSONAllocator(BufferAllocator::Get(context));

    auto alc = allocator.GetYYAlc();

    auto doc = AirportJSONCommon::CreateDocument(alc);
    auto result_obj = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, result_obj);

    auto filters_arr = yyjson_mut_arr(doc);

    for (auto &f : filters)
    {
      auto serializer = AirportJsonSerializer(doc, false, false, false);
      f->Serialize(serializer);
      yyjson_mut_arr_append(filters_arr, serializer.GetRootObject());
    }

    // With filter_pushdown enabled, an earlier pushdown pass may already have
    // moved static filters from `filters` into get.table_filters, and DuckDB
    // calls this callback again on every pass. Re-deriving them here keeps
    // json_filters (and its exactness) describing the whole static filter set
    // instead of only whatever arrived in this pass.
    //
    // LogicalGet::table_filters is keyed by the absolute table column index
    // (see CreateTableFilterSet in plan_get.cpp), while the serialized binding
    // indexes column_binding_names_by_index, i.e. positions in GetColumnIds().
    const auto &column_ids = get.GetColumnIds();
    vector<unique_ptr<Expression>> table_filter_expressions;
    for (auto &entry : get.table_filters.filters)
    {
      optional_idx position;
      for (idx_t i = 0; i < column_ids.size(); i++)
      {
        if (!column_ids[i].IsRowIdColumn() && column_ids[i].GetPrimaryIndex() == entry.first)
        {
          position = i;
          break;
        }
      }
      if (!position.IsValid() || entry.first >= get.returned_types.size())
        continue;
      BoundColumnRefExpression column(get.returned_types[entry.first],
                                      ColumnBinding(get.table_index, position.GetIndex()));
      auto expression = entry.second->ToExpression(column);
      if (!expression)
        continue;
      auto serializer = AirportJsonSerializer(doc, false, false, false);
      expression->Serialize(serializer);
      yyjson_mut_arr_append(filters_arr, serializer.GetRootObject());
      table_filter_expressions.push_back(std::move(expression));
    }

    yyjson_mut_val *column_id_names = yyjson_mut_arr(doc);
    for (auto id : get.GetColumnIds())
    {
      // So there can be a special column id specified called rowid.
      yyjson_mut_arr_add_str(doc, column_id_names, id.IsRowIdColumn() ? "rowid" : get.names[id.GetPrimaryIndex()].c_str());
    }

    yyjson_mut_obj_add_val(doc, result_obj, "filters", filters_arr);
    yyjson_mut_obj_add_val(doc, result_obj, "column_binding_names_by_index", column_id_names);
    idx_t len;
    yyjson_write_err write_error;
    auto data = yyjson_mut_val_write_opts(
        result_obj,
        AirportJSONCommon::WRITE_FLAG,
        alc, reinterpret_cast<size_t *>(&len), &write_error);

    if (data == nullptr)
    {
      throw SerializationException(
          "Failed to serialize json, perhaps the query contains invalid utf8 characters? Error %s",
          write_error.msg);
    }

    auto json_result = string(data, (size_t)len);

    auto &bind_data = bind_data_p->Cast<AirportTakeFlightBindData>();

    // json_filters and fbl_filters_exact are one snapshot of the filter set as
    // of the most recent pushdown pass, so they must be overwritten together on
    // every invocation. This callback is registered on four table functions and
    // is not idempotent-guarded, while the bind data it writes is owned by the
    // LogicalGet and read lazily at execution time. Skipping the assignment on
    // an empty filter vector would leave a previous pass's predicates in
    // json_filters after the plan stopped carrying them, and the server would
    // then filter on predicates the query no longer has.
    bind_data.json_filters = json_result;
    // all_of is vacuously true on an empty vector. Require a non-empty filter
    // set explicitly: "exact" authorizes the aggregate rewrite to clear
    // get.table_filters, which a pass that inspected no filter must never do.
    const auto is_exact = [](const unique_ptr<Expression> &filter)
    { return AirportIsExactFblFilter(*filter); };
    bind_data.fbl_filters_exact =
        bind_data.fbl_exact_filter_pushdown &&
        (!filters.empty() || !table_filter_expressions.empty()) &&
        std::all_of(filters.begin(), filters.end(), is_exact) &&
        std::all_of(table_filter_expressions.begin(), table_filter_expressions.end(), is_exact);

    // Keep normal DuckDB filters as a correctness backstop. The post-optimizer
    // aggregate rewrite removes them only after CSE has seen their distinct
    // constants; clearing here makes query09's five ranges look identical.
  }

  shared_ptr<ArrowArrayStreamWrapper> AirportProduceArrowScan(
      const AirportArrowScanFunctionData &function,
      const vector<column_t> &column_ids,
      const TableFilterSet *filters,
      atomic<double> *progress,
      std::shared_ptr<arrow::Buffer> *last_app_metadata,
      const std::shared_ptr<arrow::Schema> &schema,
      const AirportLocationDescriptor &location_descriptor,
      AirportArrowScanLocalState &local_state)
  {
    AirportArrowStreamParameters parameters(progress,
                                            last_app_metadata,
                                            schema,
                                            location_descriptor);

    auto &projected = parameters.projected_columns;
    // Preallocate space for efficiency
    projected.columns.reserve(column_ids.size());
    projected.projection_map.reserve(column_ids.size());
    projected.filter_to_col.reserve(column_ids.size());

    for (const auto col_idx : column_ids)
    {
      if (col_idx == COLUMN_IDENTIFIER_ROW_ID)
        continue;

      const auto &schema = *function.schema_root.arrow_schema.children[col_idx];
      projected.projection_map.emplace(col_idx, schema.name);
      projected.columns.emplace_back(schema.name);
      projected.filter_to_col.emplace(col_idx, col_idx);
    }

    parameters.filters = (TableFilterSet *)filters;

    return function.scanner_producer((uintptr_t)&local_state, parameters);
  }

  // static string CompressString(const string &input, const string &location, const flight::FlightDescriptor &descriptor)
  // {
  //   auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD, 1).ValueOrDie();

  //   // Estimate the maximum compressed size (usually larger than original size)
  //   int64_t max_compressed_len = codec->MaxCompressedLen(input.size(), reinterpret_cast<const uint8_t *>(input.data()));

  //   // Allocate a buffer to hold the compressed data

  //   AIRPORT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto compressed_buffer, arrow::AllocateBuffer(max_compressed_len), location, descriptor, "");

  //   // Perform the compression
  //   AIRPORT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto compressed_size,
  //                                                      codec->Compress(
  //                                                          input.size(),
  //                                                          reinterpret_cast<const uint8_t *>(input.data()),
  //                                                          max_compressed_len,
  //                                                          compressed_buffer->mutable_data()),
  //                                                      location, descriptor, "");

  //   // If you want to write the compressed data to a string
  //   std::string compressed_str(reinterpret_cast<const char *>(compressed_buffer->data()), compressed_size);
  //   return compressed_str;
  // }

  namespace
  {
    struct AirportEndpointParameters
    {
      struct Aggregate
      {
        std::string function;
        std::optional<idx_t> input_column;
        MSGPACK_DEFINE_MAP(function, input_column)
      };

      std::string json_filters;
      std::vector<idx_t> column_ids;
      bool projected_result = false;
      bool require_exact_filters = false;
      std::vector<Aggregate> aggregates;

      // The parameters to the table function, which should
      // be included in the opaque ticket data returned
      // for each endpoint.
      std::string table_function_parameters;
      std::string table_function_input_schema;

      std::string at_unit;
      std::string at_value;

      // Optional row-filter hints in the json_filters document shape. Unlike
      // json_filters they never participate in exactness: the join that
      // produced them still evaluates its own condition above the scan.
      std::string hint_filters;

      MSGPACK_DEFINE_MAP(json_filters, column_ids, projected_result, require_exact_filters, aggregates,
                         table_function_parameters, table_function_input_schema, at_unit, at_value,
                         hint_filters)
    };

    // static string BuildCompressedTicketMetadata(const string &json_filters, const vector<idx_t> &column_ids, uint32_t *uncompressed_length, const string &location, const flight::FlightDescriptor &descriptor)
    // {
    //   AirportTicketMetadataParameters params;
    //   params.json_filters = json_filters;
    //   params.column_ids = column_ids;

    //   std::stringstream packed_buffer;
    //   msgpack::pack(packed_buffer, params);

    //   auto metadata_doc_string = packed_buffer.str();
    //   *uncompressed_length = metadata_doc_string.size();
    //   auto compressed_metadata = CompressString(metadata_doc_string, location, descriptor);

    //   return compressed_metadata;
    // }

    struct AirportGetFlightEndpointsRequest
    {
      std::string descriptor;
      AirportEndpointParameters parameters;

      MSGPACK_DEFINE_MAP(descriptor, parameters)
    };

    vector<flight::FlightEndpoint> AirportGetFlightEndpoints(
        const AirportTakeFlightParameters &take_flight_params,
        const string &trace_id,
        const flight::FlightDescriptor &descriptor,
        const std::shared_ptr<flight::FlightClient> &flight_client,
        const std::string &json_filters,
        const std::vector<idx_t> &column_ids,
        bool projected_result,
        bool require_exact_filters,
        const std::vector<AirportTakeFlightBindData::FblAggregate> &aggregates,
        const std::string &table_function_parameters,
        const std::string &table_function_input_schema,
        const std::string &hint_filters)
    {
      vector<flight::FlightEndpoint> endpoints;
      arrow::flight::FlightCallOptions call_options;
      auto &server_location = take_flight_params.server_location();

      airport_add_normal_headers(call_options, take_flight_params, trace_id,
                                 descriptor);

      AirportGetFlightEndpointsRequest endpoints_request;

      AIRPORT_ASSIGN_OR_RAISE_LOCATION(
          endpoints_request.descriptor,
          descriptor.SerializeToString(),
          server_location,
          "endpoints serialize flight descriptor");

      endpoints_request.parameters.json_filters = json_filters;
      endpoints_request.parameters.column_ids = column_ids;
      endpoints_request.parameters.projected_result = projected_result;
      endpoints_request.parameters.require_exact_filters = require_exact_filters;
      for (const auto &aggregate : aggregates)
      {
        endpoints_request.parameters.aggregates.push_back({aggregate.function, aggregate.input_column});
      }
      endpoints_request.parameters.table_function_parameters = table_function_parameters;
      endpoints_request.parameters.table_function_input_schema = table_function_input_schema;
      endpoints_request.parameters.at_unit = take_flight_params.at_unit();
      endpoints_request.parameters.at_value = take_flight_params.at_value();
      endpoints_request.parameters.hint_filters = hint_filters;

      AIRPORT_MSGPACK_ACTION_SINGLE_PARAMETER(action, "endpoints", endpoints_request);

      auto serialized_endpoint_info_buffer = AirportCallAction(flight_client, call_options, action, server_location);

      std::string_view serialized_endpoint_info(reinterpret_cast<const char *>(serialized_endpoint_info_buffer->body->data()), serialized_endpoint_info_buffer->body->size());

      std::vector<std::string> serialized_endpoints;
      AIRPORT_MSGPACK_UNPACK(serialized_endpoints,
                             serialized_endpoint_info,
                             server_location,
                             "File to parse msgpack encoded endpoints");

      endpoints.reserve(serialized_endpoints.size());

      for (const auto &endpoint : serialized_endpoints)
      {
        AIRPORT_ASSIGN_OR_RAISE_LOCATION(auto deserialized_endpoint,
                                         arrow::flight::FlightEndpoint::Deserialize(endpoint),
                                         server_location,
                                         "deserialize flight endpoint");
        endpoints.push_back(std::move(deserialized_endpoint));
      }
      return endpoints;
    }
  }

  // Serializes the runtime join filters DuckDB merged into input.filters into
  // the json_filters document shape, so the server's existing parser reads
  // them. DuckDB creates these after the hash-join build side finished and
  // calls init_global only then, so the min/max and IN lists are populated.
  //
  // Every filter here is sent as a hint. The set mixes the static WHERE
  // filters (already in json_filters, and enforced again by the scan itself)
  // with optional join filters the join still enforces, so a server may drop
  // non-matching rows but no row may appear that the query would not produce
  // anyway. Bloom and uninitialized filters serialize to a TRUE constant and
  // are skipped.
  static string AirportSerializeHintFilters(ClientContext &context,
                                            const AirportTakeFlightBindData &bind_data,
                                            const TableFunctionInitInput &input)
  {
    if (!bind_data.fbl_hint_filters || !bind_data.fbl_aggregates.empty() || !input.filters ||
        input.filters->filters.empty())
    {
      return "";
    }

    auto allocator = AirportJSONAllocator(BufferAllocator::Get(context));
    auto alc = allocator.GetYYAlc();
    auto doc = AirportJSONCommon::CreateDocument(alc);
    auto result_obj = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, result_obj);
    auto filters_arr = yyjson_mut_arr(doc);

    idx_t emitted = 0;
    for (auto &entry : input.filters->filters)
    {
      const idx_t scan_index = entry.first;
      if (scan_index >= input.column_ids.size())
        continue;
      const auto column_id = input.column_ids[scan_index];
      if (column_id == COLUMN_IDENTIFIER_ROW_ID || column_id >= bind_data.return_types().size())
        continue;
      // The binding's column_index indexes column_binding_names_by_index below,
      // matching the layout AirportTakeFlightComplexFilterPushdown sends.
      BoundColumnRefExpression column(bind_data.return_types()[column_id], ColumnBinding(0, scan_index));
      auto expression = entry.second->ToExpression(column);
      if (!expression || expression->expression_class == ExpressionClass::BOUND_CONSTANT)
        continue;
      auto serializer = AirportJsonSerializer(doc, false, false, false);
      expression->Serialize(serializer);
      yyjson_mut_arr_append(filters_arr, serializer.GetRootObject());
      ++emitted;
    }
    if (emitted == 0)
      return "";

    yyjson_mut_val *column_id_names = yyjson_mut_arr(doc);
    for (auto column_id : input.column_ids)
    {
      if (column_id == COLUMN_IDENTIFIER_ROW_ID || column_id >= bind_data.return_names().size())
      {
        yyjson_mut_arr_add_str(doc, column_id_names, "rowid");
        continue;
      }
      yyjson_mut_arr_add_strcpy(doc, column_id_names, bind_data.return_names()[column_id].c_str());
    }

    yyjson_mut_obj_add_val(doc, result_obj, "filters", filters_arr);
    yyjson_mut_obj_add_val(doc, result_obj, "column_binding_names_by_index", column_id_names);
    idx_t len;
    yyjson_write_err write_error;
    auto data = yyjson_mut_val_write_opts(result_obj, AirportJSONCommon::WRITE_FLAG, alc,
                                          reinterpret_cast<size_t *>(&len), &write_error);
    if (data == nullptr)
    {
      // Hints are optional; losing them costs bytes, never correctness.
      return "";
    }
    return string(data, (size_t)len);
  }

  unique_ptr<GlobalTableFunctionState> AirportArrowScanInitGlobal(ClientContext &context,
                                                                  TableFunctionInitInput &input)
  {
    auto &bind_data = input.bind_data->CastNoConst<AirportTakeFlightBindData>();

    // Ideally this is where we call GetFlightInfo to obtain the endpoints, but
    // GetFlightInfo can't take the predicate information, so we'll need to call an
    // action called endpoints.
    //
    // FIXME: somehow the flight should be marked if it supports predicate pushdown.
    // right now I'm not sure what this is.
    //
    auto flight_client = AirportAPI::FlightClientForLocation(bind_data.server_location());

    vector<idx_t> projection_ids;
    vector<LogicalType> scanned_types;

    if (!input.projection_ids.empty())
    {
      projection_ids = input.projection_ids;
      for (const auto &col_idx : input.column_ids)
      {
        if (col_idx == COLUMN_IDENTIFIER_ROW_ID)
        {
          auto rowid_type = AirportAPI::GetRowIdType(
              context,
              bind_data.schema(),
              bind_data);
          scanned_types.emplace_back(rowid_type);
        }
        else
        {
          scanned_types.push_back(bind_data.all_types[col_idx]);
        }
      }
    }

    std::vector<idx_t> endpoint_column_ids;
    if (bind_data.fbl_aggregates.empty())
      endpoint_column_ids.assign(input.column_ids.begin(), input.column_ids.end());
    else
      endpoint_column_ids = bind_data.fbl_scan_column_ids;

    auto result = make_uniq<AirportArrowScanGlobalState>(
        AirportGetFlightEndpoints(bind_data.take_flight_params(),
                                  bind_data.trace_id(),
                                  bind_data.descriptor(),
                                  flight_client,
                                  bind_data.json_filters,
                                  endpoint_column_ids,
                                  bind_data.fbl_projection_pushdown,
                                  bind_data.require_exact_filters,
                                  bind_data.fbl_aggregates,
                                  bind_data.table_function_parameters().has_value() ? bind_data.table_function_parameters()->parameters : "",
                                  bind_data.table_function_parameters().has_value() ? bind_data.table_function_parameters()->table_input_schema : "",
                                  AirportSerializeHintFilters(context, bind_data, input)),
        projection_ids,
        scanned_types,
        input);

    // Store the total number of endpoints in the bind data so progress
    // can be reported across all endpoints.
    bind_data.set_endpoint_count(result->total_endpoints());

    return result;
  }

  double AirportTakeFlightScanProgress(ClientContext &, const FunctionData *data, const GlobalTableFunctionState *global_state)
  {
    return data->Cast<AirportTakeFlightBindData>().total_progress();
  }

  static std::vector<uint8_t> base64_decode(const std::string &base64_input)
  {
    BIO *bio, *b64;
    int decodeLen = (int)base64_input.size() * 3 / 4;
    std::vector<uint8_t> buffer(decodeLen);

    bio = BIO_new_mem_buf(base64_input.data(), (int)base64_input.length());
    b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);

    int len = BIO_read(bio, buffer.data(), (int)buffer.size());
    buffer.resize(len);
    BIO_free_all(bio);
    return buffer;
  }

  struct LocationDataContents
  {
    std::string format;
    std::string uri;

    MSGPACK_DEFINE_MAP(format, uri)
  };

  static bool
  AirportLocalStateProcessEndpoint(ClientContext &context,
                                   const TableFunctionInitInput &input,
                                   const AirportTakeFlightBindData &bind_data,
                                   AirportArrowScanGlobalState &global_state,
                                   AirportArrowScanLocalState &local_state,
                                   const flight::FlightEndpoint endpoint)
  {
    if (endpoint.locations.empty())
    {
      throw AirportFlightException(bind_data.server_location(), "No locations specified in flight endpoint");
    }

    auto &location = endpoint.locations.front();

    auto server_location = location.ToString();

    local_state.chunk_offset = 0;
    local_state.chunk = make_uniq<ArrowArrayWrapper>();
    local_state.Reset();

    if (location.scheme() == "data")
    {
      arrow::util::Uri uri;

      // Now show to we get the rest of the url.
      AIRPORT_ARROW_ASSERT_OK_LOCATION(
          uri.Parse(location.ToString()),
          server_location,
          "airport_take_flight: parsing data endpoint");

      const std::string path = uri.path();

      // Find the comma separating metadata from payload
      auto comma_pos = path.find(',');
      if (comma_pos == std::string::npos)
      {
        throw AirportFlightException(server_location, "Invaid data URI format in endpoint");
      }

      std::string media_type = path.substr(0, comma_pos); // application/msgpack;base64

      if (media_type != "application/msgpack;base64" &&
          media_type != "application/x-msgpack-duckdb-function-call;base64")
      {
        throw AirportFlightException(server_location, "Invalid media type in data URI, should be application/msgpack;base64 or application/x-msgpack-duckdb-function-call;base64");
      }

      std::string base64_payload = path.substr(comma_pos + 1); // base64 encoded data

      std::vector<uint8_t> decoded = base64_decode(base64_payload);

      if (media_type == "application/x-msgpack-duckdb-function-call;base64")
      {
        // Now deal with the msgpack stuff.
        AirportDuckDBFunctionCall function_call_data;
        AIRPORT_MSGPACK_UNPACK_CONTAINER(
            function_call_data,
            decoded,
            (&bind_data),
            "File to parse msgpack encoded DuckDB function call");

        D_ASSERT(!function_call_data.function_name.empty());
        D_ASSERT(!function_call_data.data.empty());

        auto parsed_info = AirportParseFunctionCallDetails(
            function_call_data,
            context,
            bind_data);

        auto local_scan_data = std::make_shared<AirportLocalScanData>(
            parsed_info.argument_values,
            parsed_info.named_params,
            context,
            parsed_info.func,
            bind_data.return_types(),
            bind_data.return_names(),
            *global_state.init_input());

        local_state.set_reader(local_scan_data);
      }
      else
      {
        // Now deal with the msgpack stuff.
        LocationDataContents location_data;
        AIRPORT_MSGPACK_UNPACK(location_data,
                               decoded,
                               server_location,
                               "File to parse msgpack encoded data uri");

        if (location_data.format != "ipc-stream" &&
            location_data.format != "ipc-file" &&
            location_data.format != "parquet")
        {
          throw AirportFlightException(server_location, "Unhandled data format in data URI: " + location_data.format);
        }

        // So the location can be a URL of various types, file://, s3://, gcs://

        if (location_data.uri.empty())
        {
          throw AirportFlightException(server_location, "Empty uri in data URI");
        }

        if (location_data.format == "parquet")
        {
          auto &instance = DatabaseInstance::GetDatabase(context);
          auto &parquet_scan_entry = AirportGetTableFunction(instance, "parquet_scan");
          auto &parquet_scan = parquet_scan_entry.functions.functions[0];

          // So the problem here is that we need to pass the actual return_types and return_names
          // that will be set in the output, otherwise, the output mapping is incorrect.
          auto local_scan_data = std::make_shared<AirportLocalScanData>(
              location_data.uri,
              context,
              parquet_scan,
              bind_data.return_types(),
              bind_data.return_names(),
              *global_state.init_input());

          local_state.set_reader(local_scan_data);
        }
        else if (location_data.format == "ipc-stream" || location_data.format == "ipc-file")
        {
          std::string actual_path;
          AIRPORT_ASSIGN_OR_RAISE_LOCATION(auto fs,
                                           arrow::fs::FileSystemFromUriOrPath(location_data.uri, &actual_path),
                                           server_location,
                                           "airport_take_flight: parsing data URI");

          AIRPORT_ASSIGN_OR_RAISE_LOCATION(auto input_file, fs->OpenInputFile(actual_path),
                                           server_location,
                                           "airport_take_flight: opening data URI");

          if (location_data.format == "ipc-stream")
          {
            AIRPORT_ASSIGN_OR_RAISE_LOCATION(
                auto reader,
                arrow::ipc::RecordBatchStreamReader::Open(input_file),
                server_location,
                "airport_take_flight: opening data URI")

            if (!reader->schema()->Equals(bind_data.schema()))
            {
              throw AirportFlightException(server_location, "Schema of data at" + location_data.uri + " does not match expected schema.");
            }

            local_state.set_reader(std::move(reader));
          }
          else if (location_data.format == "ipc-file")
          {
            AIRPORT_ASSIGN_OR_RAISE_LOCATION(
                auto reader,
                arrow::ipc::RecordBatchFileReader::Open(input_file),
                server_location,
                "airport_take_flight: opening data URI")

            if (!reader->schema()->Equals(bind_data.schema()))
            {
              throw AirportFlightException(server_location, "Schema of data at" + location_data.uri + " does not match expected schema.");
            }

            local_state.set_reader(std::move(reader));
          }
        }
      }
    }

    else
    {
      const auto &descriptor = bind_data.descriptor();

      arrow::flight::FlightCallOptions call_options;
      airport_add_normal_headers(call_options,
                                 bind_data.take_flight_params(),
                                 bind_data.trace_id(),
                                 descriptor);

      if (bind_data.skip_producing_result_for_update_or_delete)
      {
        // This is a special case where the result of the scan should be skipped.
        // This is useful when the scan is being used to update or delete rows.
        // For a table that doesn't actually produce row ids, so filtering cannot be applied.
        call_options.headers.emplace_back("airport-skip-producing-results", "1");
      }

      vector<string> location_errors;
      bool opened = false;
      for (const auto &candidate_location : endpoint.locations)
      {
        auto candidate_client = AirportAPI::FlightClientForLocation(bind_data.server_location());
        if (candidate_location != flight::Location::ReuseConnection())
        {
          // The per-location client cache, not a fresh Connect: a new client
          // is a new gRPC channel (TCP + HTTP/2 handshake, then teardown) for
          // every endpoint of every scan, a fixed cost that dominated small
          // scans. gRPC channels are safe for concurrent calls.
          try
          {
            candidate_client = AirportAPI::FlightClientForLocation(candidate_location.ToString());
          }
          catch (const std::exception &e)
          {
            location_errors.push_back(candidate_location.ToString() + ": connect: " + e.what());
            continue;
          }
        }

        auto stream_result = candidate_client->DoGet(call_options, endpoint.ticket);
        if (!stream_result.ok())
        {
          location_errors.push_back(candidate_location.ToString() + ": DoGet: " +
                                    stream_result.status().ToString());
          continue;
        }
        auto flight_reader = std::move(stream_result).ValueOrDie();
        AIRPORT_ASSIGN_OR_RAISE_LOCATION(
            auto output_schema,
            flight_reader->GetSchema(),
            candidate_location.ToString(),
            "DoGet output schema");
        ArrowSchemaWrapper output_schema_root;
        AIRPORT_ARROW_ASSERT_OK_LOCATION(
            ExportSchema(*output_schema, &output_schema_root.arrow_schema),
            candidate_location.ToString(),
            "export DoGet output schema");
        auto output_arrow_table = make_uniq<AirportArrowTableSchema>();
        vector<LogicalType> output_types;
        vector<string> output_names;
        idx_t output_rowid_column_index = COLUMN_IDENTIFIER_ROW_ID;
        AirportExamineSchema(context,
                             output_schema_root,
                             output_arrow_table.get(),
                             &output_types,
                             &output_names,
                             nullptr,
                             &output_rowid_column_index,
                             true);
        local_state.set_flight_output_schema(std::move(output_schema),
                                             std::move(output_arrow_table));
        local_state.set_reader(std::move(flight_reader));
        opened = true;
        break;
      }
      if (!opened)
      {
        throw AirportFlightException(
            bind_data.server_location(),
            "All advertised locations failed before opening DoGet: " + StringUtil::Join(location_errors, "; "));
      }

      // FIXME: make sure that the schema returned from the server is the same as
      // what we were expecting.

      // So the bind data won't have a stream set on it,
      // but the local state will, the prokblem is the CreateStream
      // callback doesn't have a reference to the local state.

      // Can we reuse the chunk?
    }

    if (!std::holds_alternative<std::shared_ptr<AirportLocalScanData>>(local_state.reader()))
    {
      local_state.set_stream(
          AirportProduceArrowScan(bind_data,
                                  input.column_ids,
                                  input.filters.get(),
                                  bind_data.get_progress_counter(0),
                                  // No need for the last metadata message.
                                  nullptr,
                                  local_state.flight_output_schema()
                                      ? local_state.flight_output_schema()
                                      : bind_data.schema(),
                                  bind_data,
                                  local_state));
    }
    else
    {
      // If we're using a scan function, no need to create an arrow stream.
      local_state.set_stream(nullptr);
    }

    if (bind_data.fbl_projection_pushdown &&
        local_state.flight_output_schema())
    {
      // The negotiated Flight stream is already physically narrowed and its
      // Arrow children are numbered 0..N-1. DuckDB's original column_ids still
      // refer to positions in the full table schema; passing those into
      // ArrowToDuckDB indexes beyond the narrow array and reports
      // "arrow_scan: array length mismatch".
      local_state.column_ids.clear();
      local_state.column_ids.reserve(
          local_state.flight_output_schema()->num_fields());
      for (idx_t column = 0;
           column < static_cast<idx_t>(
                        local_state.flight_output_schema()->num_fields());
           ++column)
      {
        local_state.column_ids.push_back(column);
      }
    }
    else
    {
      local_state.column_ids = input.column_ids;
    }
    local_state.filters = (TableFilterSet *)input.filters.get();

    // Projection pushdown is always enabled.
    D_ASSERT(bind_data.projection_pushdown_enabled);
    if (!input.projection_ids.empty())
    {
      local_state.all_columns.Initialize(context, global_state.scanned_types());
    }
    if (!AirportArrowScanParallelStateNext(local_state,
                                           global_state,
                                           bind_data,
                                           context))
    {
      return false;
    }
    return true;
  }

  // Compiles the table filters DuckDB pushed into this scan into one
  // conjunction over the column_ids layout (the layout of both all_columns and,
  // without projection_ids, the output chunk). Optional, dynamic and bloom
  // filters at the top level are hints the join above still enforces, so they
  // are skipped; static filters are mandatory because the planner no longer
  // keeps a residual FILTER above a scan that accepts filter pushdown.
  static unique_ptr<Expression> AirportBuildStaticFilterExpression(ClientContext &context,
                                                                    const AirportTakeFlightBindData &bind_data,
                                                                    const TableFunctionInitInput &input)
  {
    if (!input.filters || input.filters->filters.empty())
      return nullptr;

    vector<unique_ptr<Expression>> conjuncts;
    for (auto &entry : input.filters->filters)
    {
      const auto filter_type = entry.second->filter_type;
      if (filter_type == TableFilterType::OPTIONAL_FILTER || filter_type == TableFilterType::DYNAMIC_FILTER ||
          filter_type == TableFilterType::BLOOM_FILTER)
        continue;
      const idx_t scan_index = entry.first;
      if (scan_index >= input.column_ids.size())
        throw InternalException("Airport: table filter references scan column %llu of %llu", scan_index,
                                input.column_ids.size());
      const auto column_id = input.column_ids[scan_index];
      LogicalType column_type;
      if (column_id == COLUMN_IDENTIFIER_ROW_ID)
        column_type = AirportAPI::GetRowIdType(context, bind_data.schema(), bind_data);
      else if (column_id < bind_data.all_types.size())
        column_type = bind_data.all_types[column_id];
      else
        throw InternalException("Airport: table filter references unknown column %llu", column_id);
      BoundReferenceExpression column(column_type, scan_index);
      conjuncts.push_back(entry.second->ToExpression(column));
    }
    if (conjuncts.empty())
      return nullptr;

    if (conjuncts.size() == 1)
      return std::move(conjuncts[0]);
    auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
    for (auto &conjunct : conjuncts)
      conjunction->children.push_back(std::move(conjunct));
    return std::move(conjunction);
  }

  static unique_ptr<LocalTableFunctionState>
  AirportArrowScanInitLocalInternal(ClientContext &context, TableFunctionInitInput &input,
                                    GlobalTableFunctionState *global_state_p)
  {
    auto &bind_data = input.bind_data->Cast<AirportTakeFlightBindData>();
    auto &global_state = global_state_p->Cast<AirportArrowScanGlobalState>();

    auto &endpoint_opt = global_state.GetNextEndpoint();

    // If there are no endpoints, don't create a local state.
    if (!endpoint_opt)
    {
      return nullptr;
    }

    auto current_chunk = make_uniq<ArrowArrayWrapper>();
    auto result = make_uniq<AirportArrowScanLocalState>(
        std::move(current_chunk),
        context,
        input);
    result->static_filter_expression = AirportBuildStaticFilterExpression(context, bind_data, input);
    if (result->static_filter_expression)
      result->static_filter_executor = make_uniq<ExpressionExecutor>(context, *result->static_filter_expression);

    AirportLocalStateProcessEndpoint(context,
                                     input,
                                     bind_data,
                                     global_state,
                                     *result,
                                     *endpoint_opt);
    return result;
  }

  unique_ptr<LocalTableFunctionState> AirportArrowScanInitLocal(ExecutionContext &context,
                                                                TableFunctionInitInput &input,
                                                                GlobalTableFunctionState *global_state_p)
  {
    return AirportArrowScanInitLocalInternal(context.client, input, global_state_p);
  }

  static BindInfo AirportTakeFlightGetBindInfo(const optional_ptr<FunctionData> bind_data_p)
  {
    auto &bind_data = bind_data_p->Cast<AirportTakeFlightBindData>();
    // I know I'm dropping the const here, fix this later.
    AirportTableEntry *table_entry = (AirportTableEntry *)bind_data.table_entry();

    D_ASSERT(table_entry != nullptr);
    BindInfo bind_info(*table_entry);
    return bind_info;
  }

  // A token identifying this process, minted once per extension load.
  //
  // An Airport scan's bind state is only meaningful inside the process that
  // produced it, so the (de)serialize pair below refuses to revive a plan
  // anywhere else. A pid would be the obvious token but can be reused after a
  // restart, which is exactly the case that must not silently succeed.
  static uint64_t AirportPlanCopyToken()
  {
    static const uint64_t token = []()
    {
      std::random_device source;
      return ((uint64_t)source() << 32) ^ (uint64_t)source();
    }();
    return token;
  }

  // The serializer callbacks are also reachable through persistent serializers
  // such as json_serialize_plan. Airport bind state is deliberately not
  // persistable: it contains a live catalog entry and Arrow objects owned by
  // this DuckDB process. Keep an owned, one-shot snapshot for DuckDB's binary
  // in-process plan copy instead of putting a live object's address in the
  // serialized stream.
  struct AirportPlanCopySnapshot
  {
    explicit AirportPlanCopySnapshot(const AirportTakeFlightBindData &source)
        : take_flight_params(source.take_flight_params()),
          descriptor(source.descriptor()),
          schema(source.schema()),
          estimated_records(source.estimated_records()),
          table_function_parameters(source.table_function_parameters()),
          table_entry(source.table_entry()),
          json_filters(source.json_filters),
          fbl_projection_pushdown(source.fbl_projection_pushdown),
          fbl_exact_filter_pushdown(source.fbl_exact_filter_pushdown),
          fbl_filters_exact(source.fbl_filters_exact),
          fbl_hint_filters(source.fbl_hint_filters),
          fbl_partial_aggregates(source.fbl_partial_aggregates),
          require_exact_filters(source.require_exact_filters),
          fbl_aggregates(source.fbl_aggregates),
          fbl_scan_column_ids(source.fbl_scan_column_ids),
          rowid_column_index(source.rowid_column_index),
          skip_producing_result_for_update_or_delete(
              source.skip_producing_result_for_update_or_delete)
    {
    }

    AirportTakeFlightParameters take_flight_params;
    flight::FlightDescriptor descriptor;
    std::shared_ptr<arrow::Schema> schema;
    int64_t estimated_records;
    std::optional<AirportTableFunctionFlightInfoParameters> table_function_parameters;
    const AirportTableEntry *table_entry;
    string json_filters;
    bool fbl_projection_pushdown;
    bool fbl_exact_filter_pushdown;
    bool fbl_filters_exact;
    bool fbl_hint_filters;
    std::vector<std::string> fbl_partial_aggregates;
    bool require_exact_filters;
    std::vector<AirportTakeFlightBindData::FblAggregate> fbl_aggregates;
    std::vector<idx_t> fbl_scan_column_ids;
    idx_t rowid_column_index;
    bool skip_producing_result_for_update_or_delete;
  };

  static std::mutex &AirportPlanCopyMutex()
  {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<uint64_t, std::unique_ptr<AirportPlanCopySnapshot>> &
  AirportPlanCopySnapshots()
  {
    static std::unordered_map<uint64_t, std::unique_ptr<AirportPlanCopySnapshot>> snapshots;
    return snapshots;
  }

  static uint64_t AirportRegisterPlanCopy(const AirportTakeFlightBindData &source)
  {
    std::random_device random;
    std::lock_guard<std::mutex> guard(AirportPlanCopyMutex());
    auto &snapshots = AirportPlanCopySnapshots();
    uint64_t handle;
    do
    {
      handle = ((uint64_t)random() << 32) ^ (uint64_t)random();
    } while (handle == 0 || snapshots.find(handle) != snapshots.end());
    snapshots.emplace(handle, std::make_unique<AirportPlanCopySnapshot>(source));
    return handle;
  }

  static std::unique_ptr<AirportPlanCopySnapshot> AirportTakePlanCopy(uint64_t handle)
  {
    std::lock_guard<std::mutex> guard(AirportPlanCopyMutex());
    auto &snapshots = AirportPlanCopySnapshots();
    auto entry = snapshots.find(handle);
    if (entry == snapshots.end())
    {
      throw SerializationException(
          "airport_take_flight: plan-copy state is missing or has already been consumed");
    }
    auto snapshot = std::move(entry->second);
    snapshots.erase(entry);
    return snapshot;
  }

  // Airport bind state cannot be written as bytes: it holds raw pointers into
  // the local catalog, a live Arrow schema, and pushdown state negotiated with
  // the server. DuckDB nevertheless routes LogicalOperator::Copy through the
  // table function's (de)serialize pair, and that copy is entirely in-process --
  // it serializes to a MemoryStream and deserializes it immediately, with the
  // source operator alive for the whole round trip.
  //
  // Without these callbacks DuckDB falls back to re-binding the scan from
  // LogicalGet::parameters, which catalog-driven Airport scans never populate.
  // The rebind then indexes an empty argument vector and the copy dies inside
  // take_flight_bind_with_pointer. That is what made canonical TPC-DS q95 --
  // the only query of the 99 that references a CTE twice, and therefore the only
  // one whose plan gets deep-copied -- fail to run at all.
  //
  // So register an owned one-shot snapshot, carry only its opaque handle across
  // the round trip, and reject non-binary serializers before any state is
  // registered. The process token also makes a copied stream fail clearly if
  // it somehow reaches another process.
  static void AirportTakeFlightSerialize(Serializer &serializer,
                                         const optional_ptr<FunctionData> bind_data_p,
                                         const TableFunction &function)
  {
    if (bind_data_p == nullptr)
    {
      throw SerializationException(
          "airport_take_flight: cannot copy a scan that has no bind data");
    }
    if (dynamic_cast<BinarySerializer *>(&serializer) == nullptr)
    {
      throw SerializationException(
          "airport_take_flight: Airport scans only support DuckDB's in-process binary plan copy");
    }
    serializer.WriteProperty(100, "process_token", AirportPlanCopyToken());
    serializer.WriteProperty(
        101, "plan_copy_handle",
        AirportRegisterPlanCopy(bind_data_p->Cast<AirportTakeFlightBindData>()));
  }

  static unique_ptr<FunctionData> AirportTakeFlightDeserialize(Deserializer &deserializer,
                                                              TableFunction &function)
  {
    const auto process_token = deserializer.ReadProperty<uint64_t>(100, "process_token");
    const auto handle = deserializer.ReadProperty<uint64_t>(101, "plan_copy_handle");

    if (process_token != AirportPlanCopyToken())
    {
      throw SerializationException(
          "airport_take_flight: an Airport scan can only be deserialized in the process that "
          "serialized it, because its bind state references local catalog objects");
    }

    auto source = AirportTakePlanCopy(handle);
    auto &context = deserializer.Get<ClientContext &>();

    // AirportTakeFlightBindWithFlightDescriptor ignores its bind input entirely
    // -- every value it needs is passed explicitly -- so an empty one is enough
    // to satisfy the signature.
    vector<Value> parameters;
    named_parameter_map_t named_parameters;
    vector<LogicalType> input_table_types;
    vector<string> input_table_names;
    TableFunctionRef empty_ref;
    TableFunctionBindInput bind_input(parameters, named_parameters, input_table_types,
                                      input_table_names, function.function_info.get(),
                                      nullptr, function, empty_ref);

    vector<LogicalType> return_types;
    vector<string> names;
    // Passing the source's schema keeps this off the network: a non-null schema
    // skips the GetFlightInfo round trip, so copying a plan costs no RPC.
    auto result = AirportTakeFlightBindWithFlightDescriptor(
        source->take_flight_params,
        source->descriptor,
        context,
        bind_input,
        return_types,
        names,
        source->schema,
        source->estimated_records,
        source->table_function_parameters,
        source->table_entry);

    auto &copy = result->Cast<AirportTakeFlightBindData>();

    // State the bind itself cannot recompute: filters, projections and
    // aggregates negotiated during optimization. DuckDB deep-copies plans both
    // before and after filter pushdown (CTE inlining runs twice), so a copy that
    // dropped these would silently lose pushdown rather than fail. Any new
    // mutable field on AirportTakeFlightBindData belongs here too.
    copy.json_filters = source->json_filters;
    copy.fbl_projection_pushdown = source->fbl_projection_pushdown;
    copy.fbl_exact_filter_pushdown = source->fbl_exact_filter_pushdown;
    copy.fbl_filters_exact = source->fbl_filters_exact;
    copy.fbl_hint_filters = source->fbl_hint_filters;
    copy.fbl_partial_aggregates = source->fbl_partial_aggregates;
    copy.require_exact_filters = source->require_exact_filters;
    copy.fbl_aggregates = source->fbl_aggregates;
    copy.fbl_scan_column_ids = source->fbl_scan_column_ids;
    copy.rowid_column_index = source->rowid_column_index;
    copy.skip_producing_result_for_update_or_delete =
        source->skip_producing_result_for_update_or_delete;

    return result;
  }

  void AirportAddTakeFlightFunction(ExtensionLoader &loader)
  {
    auto take_flight_function_set = TableFunctionSet("airport_take_flight");

    auto take_flight_function_with_descriptor = TableFunction(
        "airport_take_flight",
        {LogicalType::VARCHAR, LogicalType::ANY},
        AirportTakeFlight,
        take_flight_bind,
        AirportArrowScanInitGlobal,
        AirportArrowScanInitLocal);

    take_flight_function_with_descriptor.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_function_with_descriptor.named_parameters["secret"] = LogicalType::VARCHAR;
    take_flight_function_with_descriptor.named_parameters["ticket"] = LogicalType::BLOB;
    take_flight_function_with_descriptor.named_parameters["headers"] = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
    take_flight_function_with_descriptor.named_parameters["at_unit"] = LogicalType::VARCHAR;
    take_flight_function_with_descriptor.named_parameters["at_value"] = LogicalType::ANY;

    take_flight_function_with_descriptor.pushdown_complex_filter = AirportTakeFlightComplexFilterPushdown;

    take_flight_function_with_descriptor.cardinality = AirportTakeFlightCardinality;
    take_flight_function_with_descriptor.to_string = AirportTakeFlightToString;
    //    take_flight_function_with_descriptor.get_batch_index = nullptr;
    take_flight_function_with_descriptor.projection_pushdown = true;
    // Filter pushdown lets DuckDB's join-filter optimizer target Airport scans
    // (see AirportSerializeHintFilters). Pushed static filters are enforced
    // by the scan itself (AirportBuildStaticFilterExpression). filter_prune
    // stays off: with it, multi-endpoint scans (MAX_ENDPOINTS > 1) produced
    // chunks narrower than the scan's column layout (TPC-DS q4 at SF1).
    take_flight_function_with_descriptor.filter_pushdown = true;
    take_flight_function_with_descriptor.filter_prune = false;
    take_flight_function_with_descriptor.table_scan_progress = AirportTakeFlightScanProgress;
    // Required for LogicalOperator::Copy; see AirportTakeFlightSerialize.
    take_flight_function_with_descriptor.serialize = AirportTakeFlightSerialize;
    take_flight_function_with_descriptor.deserialize = AirportTakeFlightDeserialize;
    take_flight_function_set.AddFunction(take_flight_function_with_descriptor);

    auto take_flight_function_with_pointer = TableFunction(
        "airport_take_flight",
        {LogicalType::POINTER, LogicalType::POINTER, LogicalType::VARCHAR},
        AirportTakeFlight,
        take_flight_bind_with_pointer,
        AirportArrowScanInitGlobal,
        AirportArrowScanInitLocal);

    take_flight_function_with_pointer.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_function_with_pointer.named_parameters["secret"] = LogicalType::VARCHAR;
    take_flight_function_with_pointer.named_parameters["at_unit"] = LogicalType::VARCHAR;
    take_flight_function_with_pointer.named_parameters["at_value"] = LogicalType::ANY;
    take_flight_function_with_pointer.pushdown_complex_filter = AirportTakeFlightComplexFilterPushdown;

    // Add support for optional named paraemters that would be appended to the descriptor
    // of the flight, ideally parameters would be JSON encoded.

    take_flight_function_with_pointer.cardinality = AirportTakeFlightCardinality;
    //    take_flight_function_with_pointer.get_batch_index = nullptr;
    take_flight_function_with_pointer.projection_pushdown = true;
    // Filter pushdown lets DuckDB's join-filter optimizer target Airport scans
    // (see AirportSerializeHintFilters). Pushed static filters are enforced
    // by the scan itself (AirportBuildStaticFilterExpression). filter_prune
    // stays off: with it, multi-endpoint scans (MAX_ENDPOINTS > 1) produced
    // chunks narrower than the scan's column layout (TPC-DS q4 at SF1).
    take_flight_function_with_pointer.filter_pushdown = true;
    take_flight_function_with_pointer.filter_prune = false;
    take_flight_function_with_pointer.table_scan_progress = AirportTakeFlightScanProgress;
    take_flight_function_with_pointer.statistics = AirportTakeFlightStatistics;
    take_flight_function_with_pointer.get_bind_info = AirportTakeFlightGetBindInfo;
    take_flight_function_with_pointer.to_string = AirportTakeFlightToString;
    // Required for LogicalOperator::Copy; see AirportTakeFlightSerialize. This
    // is the variant catalog tables bind through, and the one canonical TPC-DS
    // q95 used to fail on.
    take_flight_function_with_pointer.serialize = AirportTakeFlightSerialize;
    take_flight_function_with_pointer.deserialize = AirportTakeFlightDeserialize;

    take_flight_function_set.AddFunction(take_flight_function_with_pointer);

    loader.RegisterFunction(take_flight_function_set);
  }

  std::string AirportNameForField(const string &name, idx_t col_idx)
  {
    if (name.empty())
    {
      return string("v") + to_string(col_idx);
    }
    return name;
  }
}
