#include "airport_optimizer.hpp"

#include "airport_extension.hpp"
#include "airport_flight_stream.hpp"
#include "airport_schema_utils.hpp"
#include "airport_take_flight.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "storage/airport_catalog.hpp"
#include "storage/airport_table_entry.hpp"

namespace duckdb {
namespace {
static void MarkAirportTakeFlightAsSkipProducing(unique_ptr<LogicalOperator> &op) {
	auto markAirportTakeFlightSkipResults = [](LogicalGet &get) {
		if (get.function.name == "airport_take_flight") {
			auto &bind_data = get.bind_data->Cast<AirportTakeFlightBindData>();
			bind_data.skip_producing_result_for_update_or_delete = true;
		}
	};

	reference<LogicalOperator> child = *op->children[0];
	if (child.get().type == LogicalOperatorType::LOGICAL_GET) {
		markAirportTakeFlightSkipResults(child.get().Cast<LogicalGet>());
	} else if (child.get().type == LogicalOperatorType::LOGICAL_FILTER ||
	           child.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		for (auto &child_node : op->children) {
			MarkAirportTakeFlightAsSkipProducing(child_node);
		}
	} else {
		throw NotImplementedException("Unsupported child type for LogicalUpdate type is" +
		                              LogicalOperatorToString(child.get().type));
	}
}

void OptimizeAirportUpdate(unique_ptr<LogicalOperator> &op) {
	if (op->type != LogicalOperatorType::LOGICAL_UPDATE)
		return;

	auto &update = op->Cast<LogicalUpdate>();
	auto &airport_table = update.table.Cast<AirportTableEntry>();

	// If the table produced rowids we cannot optimize it.
	if (airport_table.GetRowIdType() != LogicalType::SQLNULL)
		return;

	MarkAirportTakeFlightAsSkipProducing(op);
}

void OptimizeAirportDelete(unique_ptr<LogicalOperator> &op) {
	if (op->type != LogicalOperatorType::LOGICAL_DELETE)
		return;

	auto &del = op->Cast<LogicalDelete>();
	auto &airport_table = del.table.Cast<AirportTableEntry>();

	// If the table produced rowids we cannot optimize it.
	if (airport_table.GetRowIdType() != LogicalType::SQLNULL)
		return;

	MarkAirportTakeFlightAsSkipProducing(op);
}

unique_ptr<BoundAggregateExpression> BindAggregate(ClientContext &context, const string &name,
                                                   unique_ptr<Expression> argument) {
	vector<unique_ptr<Expression>> arguments;
	vector<LogicalType> types;
	if (argument) {
		types.push_back(argument->return_type);
		arguments.push_back(std::move(argument));
	}
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<AggregateFunctionCatalogEntry>(context, DEFAULT_SCHEMA, name);
	const auto function = entry.functions.GetFunctionByArguments(context, types);
	FunctionBinder binder(context);
	return binder.BindAggregateFunction(function, std::move(arguments));
}

bool SupportsAggregate(const AirportTakeFlightBindData &bind_data, const string &name) {
	return std::find(bind_data.fbl_partial_aggregates.begin(), bind_data.fbl_partial_aggregates.end(), name) !=
	       bind_data.fbl_partial_aggregates.end();
}

bool IsSupportedAggregateInput(const LogicalType &type) {
	return type.IsIntegral() || type.IsFloating() || type.id() == LogicalTypeId::DECIMAL;
}

LogicalType PartialSumType(const LogicalType &input) {
	if (input.IsFloating()) {
		return LogicalType::DOUBLE;
	}
	if (input.id() == LogicalTypeId::DECIMAL) {
		return LogicalType::DECIMAL(38, DecimalType::GetScale(input));
	}
	return LogicalType::DECIMAL(38, 0);
}

std::shared_ptr<arrow::DataType> PartialSumArrowType(const std::shared_ptr<arrow::DataType> &input) {
	if (arrow::is_floating(input->id())) {
		return arrow::float64();
	}
	if (input->id() == arrow::Type::DECIMAL128) {
		const auto &decimal = arrow::internal::checked_cast<const arrow::Decimal128Type &>(*input);
		return arrow::decimal128(38, decimal.scale());
	}
	return arrow::decimal128(38, 0);
}

unique_ptr<Expression> BindDivision(ClientContext &context, unique_ptr<Expression> left, unique_ptr<Expression> right) {
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	ErrorData error;
	FunctionBinder binder(context);
	auto result = binder.BindScalarFunction(DEFAULT_SCHEMA, "/", std::move(children), error);
	if (!result) {
		error.Throw();
	}
	return result;
}

bool OptimizeAirportAggregate(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &op) {
	if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &aggregate = op->Cast<LogicalAggregate>();
	if (!aggregate.groups.empty() || !aggregate.grouping_functions.empty() || aggregate.children.size() != 1 ||
	    (aggregate.children[0]->type != LogicalOperatorType::LOGICAL_GET &&
	     aggregate.children[0]->type != LogicalOperatorType::LOGICAL_FILTER)) {
		return false;
	}
	LogicalFilter *residual_filter = nullptr;
	LogicalOperator *scan_child = aggregate.children[0].get();
	if (scan_child->type == LogicalOperatorType::LOGICAL_FILTER) {
		residual_filter = &scan_child->Cast<LogicalFilter>();
		if (residual_filter->children.size() != 1 ||
		    residual_filter->children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}
		scan_child = residual_filter->children[0].get();
	}
	auto &get = scan_child->Cast<LogicalGet>();
	if (get.function.name != "airport_take_flight") {
		return false;
	}
	auto &bind_data = get.bind_data->Cast<AirportTakeFlightBindData>();
	// A rewritten plan is Projection -> Aggregate(sum over partials) -> Get, and
	// that shape re-matches every gate below: sum over the DECIMAL(38, 0) partial
	// passes IsSupportedAggregateInput. Without this guard a second optimizer
	// pass over an already-rewritten plan would build partials of partials.
	if (!bind_data.fbl_aggregates.empty()) {
		return false;
	}
	if (!bind_data.fbl_projection_pushdown || !bind_data.fbl_exact_filter_pushdown) {
		return false;
	}
	if (!get.table_filters.filters.empty() && !bind_data.fbl_filters_exact) {
		return false;
	}
	if (residual_filter && !bind_data.fbl_filters_exact) {
		return false;
	}

	const auto original_column_ids = get.GetColumnIds();
	std::vector<AirportTakeFlightBindData::FblAggregate> specs;
	std::vector<idx_t> source_column_ids;
	vector<LogicalType> partial_types;
	vector<string> partial_names;
	std::vector<std::shared_ptr<arrow::Field>> partial_fields;
	vector<LogicalType> original_result_types;

	for (idx_t i = 0; i < aggregate.expressions.size(); ++i) {
		auto &expression = aggregate.expressions[i];
		if (expression->expression_class != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &bound = expression->Cast<BoundAggregateExpression>();
		string function = bound.function.name;
		if (function == "count_star") {
			function = "count";
		}
		if ((function != "count" && function != "sum" && function != "avg") ||
		    !SupportsAggregate(bind_data, function) || bound.IsDistinct() || bound.filter || bound.order_bys ||
		    bound.children.size() > 1) {
			return false;
		}

		std::optional<idx_t> physical_column;
		LogicalType input_type;
		std::shared_ptr<arrow::DataType> input_arrow_type;
		if (!bound.children.empty()) {
			if (bound.children[0]->expression_class != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			const auto &column = bound.children[0]->Cast<BoundColumnRefExpression>();
			if (column.binding.table_index != get.table_index ||
			    column.binding.column_index >= original_column_ids.size()) {
				return false;
			}
			const auto physical = original_column_ids[column.binding.column_index].GetPrimaryIndex();
			if (physical == COLUMN_IDENTIFIER_ROW_ID ||
			    physical >= static_cast<idx_t>(bind_data.schema()->num_fields())) {
				return false;
			}
			physical_column = physical;
			input_type = column.return_type;
			input_arrow_type = bind_data.schema()->field(physical)->type();
			if ((function == "sum" || function == "avg") && !IsSupportedAggregateInput(input_type)) {
				return false;
			}
			if (std::find(source_column_ids.begin(), source_column_ids.end(), physical) == source_column_ids.end()) {
				source_column_ids.push_back(physical);
			}
		} else if (function != "count") {
			return false;
		}

		specs.push_back({function, physical_column});
		original_result_types.push_back(bound.return_type);
		const auto prefix = "__fbl_a" + std::to_string(i);
		if (function == "count") {
			partial_types.push_back(LogicalType::BIGINT);
			partial_names.push_back(prefix + "_count");
			partial_fields.push_back(arrow::field(prefix + "_count", arrow::int64(), false));
		} else {
			const auto sum_type = PartialSumType(input_type);
			partial_types.push_back(sum_type);
			partial_names.push_back(prefix + "_sum");
			partial_fields.push_back(arrow::field(prefix + "_sum", PartialSumArrowType(input_arrow_type), true));
			if (function == "avg") {
				partial_types.push_back(LogicalType::BIGINT);
				partial_names.push_back(prefix + "_count");
				partial_fields.push_back(arrow::field(prefix + "_count", arrow::int64(), false));
			}
		}
	}

	// fbl_filters_exact was computed over the vector handed to the filter-pushdown
	// callback. This residual filter node is a different object, and only the
	// serialized json_filters travels to the server, so prove these specific
	// expressions exact before discarding them. Checked last so the common
	// non-matching plan pays nothing for it.
	if (residual_filter &&
	    !std::all_of(residual_filter->expressions.begin(), residual_filter->expressions.end(),
	                 [](const unique_ptr<Expression> &expression) { return AirportIsExactFblFilter(*expression); })) {
		return false;
	}

	bind_data.fbl_aggregates = std::move(specs);
	bind_data.fbl_scan_column_ids = std::move(source_column_ids);
	bind_data.require_exact_filters = true;
	get.table_filters.filters.clear();
	if (residual_filter) {
		// This extension runs after DuckDB's common-subexpression pass. Keeping
		// the residual until now prevents query09's five BETWEEN ranges from
		// being deduplicated as identical Airport scans.
		//
		// Move the scan out to a local first: residual_filter points into
		// aggregate.children[0], so assigning directly would destroy the filter
		// while reading through it. That ordering happens to be safe, but only
		// because unique_ptr::operator= releases the source before deleting the
		// old pointee. Do not rely on it.
		auto scan = std::move(residual_filter->children[0]);
		residual_filter = nullptr;
		aggregate.children[0] = std::move(scan);
	}
	bind_data.set_schema(arrow::schema(std::move(partial_fields)));
	bind_data.arrow_table = AirportArrowTableSchema();
	bind_data.all_types.clear();
	vector<string> examined_names;
	AirportExamineSchema(input.context, bind_data.schema_root, &bind_data.arrow_table, &bind_data.all_types,
	                     &examined_names, nullptr, &bind_data.rowid_column_index, true);
	bind_data.set_types_and_names(bind_data.all_types, examined_names);

	get.returned_types = partial_types;
	get.names = partial_names;
	vector<ColumnIndex> partial_column_ids;
	for (idx_t i = 0; i < partial_types.size(); ++i) {
		partial_column_ids.emplace_back(i);
	}
	get.SetColumnIds(std::move(partial_column_ids));
	get.projection_ids.clear();

	vector<unique_ptr<Expression>> combine_aggregates;
	vector<unique_ptr<Expression>> final_values;
	const auto final_table_index = aggregate.aggregate_index;
	const auto combine_table_index = input.optimizer.binder.GenerateTableIndex();
	idx_t partial_index = 0;
	idx_t combine_index = 0;
	for (idx_t i = 0; i < bind_data.fbl_aggregates.size(); ++i) {
		const auto &spec = bind_data.fbl_aggregates[i];
		const auto sum_partial_index = partial_index++;
		auto partial_ref = make_uniq<BoundColumnRefExpression>(partial_types[sum_partial_index],
		                                                       ColumnBinding(get.table_index, sum_partial_index));
		auto combined_sum = BindAggregate(input.context, "sum", std::move(partial_ref));
		const auto sum_result_type = combined_sum->return_type;
		combine_aggregates.push_back(std::move(combined_sum));

		unique_ptr<Expression> final_expression =
		    make_uniq<BoundColumnRefExpression>(sum_result_type, ColumnBinding(combine_table_index, combine_index++));
		if (spec.function == "avg") {
			const auto count_partial_index = partial_index++;
			auto count_ref = make_uniq<BoundColumnRefExpression>(partial_types[count_partial_index],
			                                                     ColumnBinding(get.table_index, count_partial_index));
			auto combined_count = BindAggregate(input.context, "sum", std::move(count_ref));
			const auto count_type = combined_count->return_type;
			combine_aggregates.push_back(std::move(combined_count));
			auto final_count =
			    make_uniq<BoundColumnRefExpression>(count_type, ColumnBinding(combine_table_index, combine_index++));
			final_expression = BindDivision(
			    input.context,
			    BoundCastExpression::AddCastToType(input.context, std::move(final_expression), LogicalType::DOUBLE),
			    BoundCastExpression::AddCastToType(input.context, std::move(final_count), LogicalType::DOUBLE));
		}
		final_values.push_back(
		    BoundCastExpression::AddCastToType(input.context, std::move(final_expression), original_result_types[i]));
	}

	aggregate.aggregate_index = combine_table_index;
	aggregate.expressions = std::move(combine_aggregates);
	auto projection = make_uniq<LogicalProjection>(final_table_index, std::move(final_values));
	projection->AddChild(std::move(op));
	op = std::move(projection);
	return true;
}

void OptimizeAirportAggregates(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		OptimizeAirportAggregates(input, child);
	}
	OptimizeAirportAggregate(input, op);
}
} // namespace

void AirportOptimizer::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	OptimizeAirportUpdate(plan);
	OptimizeAirportDelete(plan);
	OptimizeAirportAggregates(input, plan);
}
} // namespace duckdb
