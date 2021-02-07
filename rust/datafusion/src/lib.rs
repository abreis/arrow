// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#![warn(missing_docs)]
// Clippy lints, some should be disabled incrementally
#![allow(
    clippy::float_cmp,
    clippy::module_inception,
    clippy::new_without_default,
    clippy::type_complexity
)]

//! DataFusion is an extensible query execution framework that uses
//! [Apache Arrow](https://arrow.apache.org) as its in-memory format.
//!
//! DataFusion supports both an SQL and a DataFrame API for building logical query plans
//! as well as a query optimizer and execution engine capable of parallel execution
//! against partitioned data sources (CSV and Parquet) using threads.
//!
//! Below is an example of how to execute a query against a CSV using [`DataFrame`]s:
//!
//! ```rust
//! # use datafusion::prelude::*;
//! # use datafusion::error::Result;
//! # use arrow::record_batch::RecordBatch;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<()> {
//! let mut ctx = ExecutionContext::new();
//!
//! // create the dataframe
//! let df = ctx.read_csv("tests/example.csv", CsvReadOptions::new())?;
//!
//! // create a plan
//! let df = df.filter(col("a").lt_eq(col("b")))?
//!            .aggregate(&[col("a")], &[min(col("b"))])?
//!            .limit(100)?;
//!
//! // execute the plan
//! let results: Vec<RecordBatch> = df.collect().await?;
//! # Ok(())
//! # }
//! ```
//!
//! and how to execute a query against a CSV using SQL:
//!
//! ```
//! # use datafusion::prelude::*;
//! # use datafusion::error::Result;
//! # use arrow::record_batch::RecordBatch;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<()> {
//! let mut ctx = ExecutionContext::new();
//!
//! ctx.register_csv("example", "tests/example.csv", CsvReadOptions::new())?;
//!
//! // create a plan
//! let df = ctx.sql("SELECT a, MIN(b) FROM example GROUP BY a LIMIT 100")?;
//!
//! // execute the plan
//! let results: Vec<RecordBatch> = df.collect().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Parse, Plan, Optimize, Execute
//!
//! DataFusion is a fully fledged query engine capable of performing complex operations.
//! Specifically, when DataFusion receives an SQL query, there are different steps
//! that it passes through until a result is obtained. Broadly, they are:
//!
//! 1. The string is parsed to an Abstract syntax tree (AST) using [sqlparser](https://docs.rs/sqlparser/).
//! 2. The planner [`SqlToRel`] converts logical expressions on the AST to logical expressions [`Expr`]s
//! 3. The planner [`SqlToRel`] converts logical nodes on the AST to a [`LogicalPlan`].
//! 4. [`OptimizerRule`]s are applied to the [`LogicalPlan`] to optimize it.
//! 5. The [`LogicalPlan`] is converted to an [`ExecutionPlan`] by a [`PhysicalPlanner`]
//! 6. The [`ExecutionPlan`] is executed against data through the [`ExecutionContext`]
//!
//! With a [`DataFrame`] API, steps 1-3 are not used as the DataFrame builds the [`LogicalPlan`] directly.
//!
//! Phases 1-5 are typically cheap when compared to phase 6, and thus DataFusion puts a
//! lot of effort to ensure that phase 6 runs efficiently and without errors.
//!
//! DataFusion's planning is divided in two main parts: logical planning and physical planning.
//!
//! ### Logical plan
//!
//! Logical planning yields [logical plans](logical_plan::LogicalPlan) and [logical expressions](logical_plan::Expr).
//! These are [`Schema`]-aware traits that represent statements whose result is independent of how it should physically be executed.
//!
//! A [`LogicalPlan`] is a Direct Acyclic graph of other [`LogicalPlan`]s and each node contains logical expressions ([`Expr`]s).
//! All of these are located in [`logical_plan`].
//!
//! ### Physical plan
//!
//! A Physical plan ([`ExecutionPlan`]) is a plan that can be executed against data.
//! Contrarily to a logical plan, the physical plan has concrete information about how the calculation
//! should be performed (e.g. what Rust functions are used) and how data should be loaded into memory.
//!
//! [`ExecutionPlan`] uses the Arrow format as its in-memory representation of data, through the [arrow] crate.
//! We recommend going through [its documentation](arrow) for details on how the data is physically represented.
//!
//! An [`ExecutionPlan`] is composed by nodes (implement the trait [`ExecutionPlan`]),
//! and each node is composed by physical expressions ([`PhysicalExpr`])
//! or aggregate expressions ([`AggregateExpr`]).
//! All of these are located in the module [`physical_plan`].
//!
//! Broadly speaking,
//!
//! * an [`ExecutionPlan`] receives a partition number and asynchronosly returns
//!   an iterator over [`RecordBatch`]
//!   (a node-specific struct that implements [`RecordBatchReader`])
//! * a [`PhysicalExpr`] receives a [`RecordBatch`]
//!   and returns an [`Array`]
//! * an [`AggregateExpr`] receives [`RecordBatch`]es
//!   and returns a [`RecordBatch`] of a single row<sup>†</sup>
//!
//! *<sup>†</sup> technically, it aggregates the results on each partition and then merges the results into a single partition*
//!
//! The following physical nodes are currently implemented:
//!
//! * Projection: [`ProjectionExec`]
//! * Filter: [`FilterExec`]
//! * Hash and Grouped aggregations: [`HashAggregateExec`]
//! * Sort: [`SortExec`]
//! * Merge (partitions): [`MergeExec`]
//! * Limit: [`LocalLimitExec`] and [`GlobalLimitExec`]
//! * Scan a CSV: [`CsvExec`]
//! * Scan a Parquet: [`ParquetExec`]
//! * Scan from memory: [`MemoryExec`]
//! * Explain the plan: [`ExplainExec`]
//!
//! ## Customize
//!
//! DataFusion allows users to
//! * extend the planner to use user-defined logical and physical nodes ([`QueryPlanner`])
//! * declare and use user-defined scalar functions ([`ScalarUDF`])
//! * declare and use user-defined aggregate functions ([`AggregateUDF`])
//!
//! You can find examples of each of them in examples section.
//!
//! [`AggregateExpr`]: physical_plan::AggregateExpr
//! [`AggregateExpr`]: physical_plan::AggregateExpr
//! [`AggregateUDF`]: physical_plan::udaf::AggregateUDF
//! [`Array`]: arrow::array::Array
//! [`CsvExec`]: physical_plan::csv::CsvExec
//! [`DataFrame`]: dataframe::DataFrame
//! [`ExecutionContext`]: execution::context::ExecutionContext
//! [`ExecutionPlan`]: physical_plan::ExecutionPlan
//! [`ExplainExec`]: physical_plan::explain::ExplainExec
//! [`Expr`]: logical_plan::Expr
//! [`FilterExec`]: physical_plan::filter::FilterExec
//! [`GlobalLimitExec`]: physical_plan::limit::GlobalLimitExec
//! [`HashAggregateExec`]: physical_plan::hash_aggregate::HashAggregateExec
//! [`LocalLimitExec`]: physical_plan::limit::LocalLimitExec
//! [`LogicalPlan`]: logical_plan::LogicalPlan
//! [`logical_plan`]: logical_plan
//! [`MemoryExec`]: physical_plan::memory::MemoryExec
//! [`MergeExec`]: physical_plan::merge::MergeExec
//! [`OptimizerRule`]: optimizer::optimizer::OptimizerRule
//! [`ParquetExec`]: physical_plan::parquet::ParquetExec
//! [`PhysicalExpr`]: physical_plan::PhysicalExpr
//! [`PhysicalPlanner`]: physical_plan::PhysicalPlanner
//! [`physical_plan`]: physical_plan
//! [`ProjectionExec`]: physical_plan::projection::ProjectionExec
//! [`QueryPlanner`]: execution::context::QueryPlanner
//! [`RecordBatchReader`]: arrow::record_batch::RecordBatchReader
//! [`RecordBatch`]: arrow::record_batch::RecordBatch
//! [`ScalarUDF`]: physical_plan::udf::ScalarUDF
//! [`Schema`]: arrow::datatypes::Schema
//! [`SortExec`]: physical_plan::sort::SortExec
//! [`SqlToRel`]: sql::planner::SqlToRel

extern crate arrow;
extern crate sqlparser;

pub mod dataframe;
pub mod datasource;
pub mod error;
pub mod execution;
pub mod logical_plan;
pub mod optimizer;
pub mod physical_plan;
pub mod prelude;
pub mod scalar;
pub mod sql;
pub mod variable;

#[cfg(test)]
pub mod test;
