#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/enums/access_mode.hpp"
#include "storage/airport_schema_set.hpp"

namespace duckdb
{
  class AirportSchemaEntry;

  struct AirportAttachParameters
  {
    AirportAttachParameters(const string &location, const string &auth_token, const string &secret_name, const string &criteria,
                            idx_t max_endpoints, bool fbl_pushdown_enabled,
                            bool fbl_hint_filters_enabled = true, const string &ipc_compression = "")
        : location_(location), auth_token_(auth_token), secret_name_(secret_name), criteria_(criteria),
          max_endpoints_(max_endpoints), fbl_pushdown_enabled_(fbl_pushdown_enabled),
          fbl_hint_filters_enabled_(fbl_hint_filters_enabled), ipc_compression_(ipc_compression)
    {
    }

    const string &location() const
    {
      return location_;
    }

    const string &auth_token() const
    {
      return auth_token_;
    }

    const string &secret_name() const
    {
      return secret_name_;
    }

    const string &criteria() const
    {
      return criteria_;
    }

    idx_t max_endpoints() const
    {
      return max_endpoints_;
    }

    bool fbl_pushdown_enabled() const
    {
      return fbl_pushdown_enabled_;
    }

    bool fbl_hint_filters_enabled() const
    {
      return fbl_hint_filters_enabled_;
    }

    const string &ipc_compression() const
    {
      return ipc_compression_;
    }

  private:
    // The location of the flight server.
    string location_;
    // The authorization token to use.
    string auth_token_;
    // The name of the secret to use
    string secret_name_;
    // The criteria to pass to the flight server when listing flights.
    string criteria_;
    // Maximum number of Flight endpoints requested for each catalog scan.
    idx_t max_endpoints_;
    // Allows immediate rollback and benchmark A/B runs without changing the
    // server. Capability checks still gate every individual optimization.
    bool fbl_pushdown_enabled_;
    // Sends DuckDB's runtime join filters (min/max, IN lists) to the server as
    // optional scan hints. Separate from FBL_PUSHDOWN so the hints can be A/B
    // measured on their own; the server's capability bit still gates them.
    bool fbl_hint_filters_enabled_;
    // Requested DoGet IPC body compression ("" = none, "lz4", "zstd"),
    // forwarded as x-fbl-ipc-compression. Arrow decompresses transparently.
    string ipc_compression_;
  };

  class AirportClearCacheFunction : public TableFunction
  {
  public:
    AirportClearCacheFunction();

    static void ClearCacheOnSetting(ClientContext &context, SetScope scope, Value &parameter);
  };

  class AirportCatalog : public Catalog
  {
  public:
    explicit AirportCatalog(AttachedDatabase &db_p, const string &internal_name, AccessMode access_mode,
                            AirportAttachParameters attach_params);
    ~AirportCatalog() override;

  public:
    void Initialize(bool load_builtin) override;
    string GetCatalogType() override
    {
      return "airport";
    }

    string GetDefaultSchema() const override;

    bool SupportsTimeTravel() const override
    {
      return true;
    }

    optional_ptr<CatalogEntry> CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) override;

    void ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) override;

    optional_ptr<SchemaCatalogEntry> LookupSchema(CatalogTransaction transaction,
                                                  const EntryLookupInfo &schema_lookup,
                                                  OnEntryNotFound if_not_found) override;

    PhysicalOperator &PlanCreateTableAs(ClientContext &context, PhysicalPlanGenerator &planner,
                                        LogicalCreateTable &op, PhysicalOperator &plan) override;
    PhysicalOperator &PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
                                 optional_ptr<PhysicalOperator> plan) override;
    PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
                                 PhysicalOperator &plan) override;

    PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
                                 PhysicalOperator &plan) override;

    unique_ptr<LogicalOperator> BindCreateIndex(Binder &binder, CreateStatement &stmt, TableCatalogEntry &table,
                                                unique_ptr<LogicalOperator> plan) override;

    DatabaseSize GetDatabaseSize(ClientContext &context) override;

    //! Whether or not this is an in-memory database
    bool InMemory() override;
    string GetDBPath() override;

    void ClearCache();

    optional_idx GetCatalogVersion(ClientContext &context) override;

    void SetLoadedCatalogVersion(AirportGetCatalogVersionResult &result)
    {
      loaded_catalog_version = result;
    }

    std::optional<string> GetTransactionIdentifier();

    const string &internal_name() const
    {
      return internal_name_;
    }

    const std::shared_ptr<AirportAttachParameters> &attach_parameters() const
    {
      return attach_parameters_;
    }

    const AccessMode &access_mode() const
    {
      return access_mode_;
    }

  private:
    void DropSchema(ClientContext &context, DropInfo &info) override;

  private:
    // Track what version of the catalog has been loaded.
    std::optional<AirportGetCatalogVersionResult> loaded_catalog_version = std::nullopt;

    std::shared_ptr<arrow::flight::FlightClient> flight_client_;
    AccessMode access_mode_;
    std::shared_ptr<AirportAttachParameters> attach_parameters_;
    string internal_name_;
    AirportSchemaSet schemas;
    string default_schema_;
  };
}
