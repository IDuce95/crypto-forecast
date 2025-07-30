
import findspark
import os
from typing import Dict, List, Optional, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lag, lead, avg, stddev, min as spark_min, max as spark_max,
    window, unix_timestamp, from_unixtime, date_format,
    collect_list, struct, sum as spark_sum, count, desc,
    regexp_replace, split, explode, array, lit, concat_ws,
    percentile_approx, corr, skewness, kurtosis
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, BooleanType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    findspark.init()
except Exception as e:
    logging.warning(f"Findspark init failed: {e}")

from .logger_manager import LoggerManager

class PySparkManager:

    def __init__(self, app_name: str = "CryptoForecastingApp",
                 master: str = "local[*]",
                 executor_memory: str = "4g",
                 driver_memory: str = "2g",
                 executor_cores: int = 2):
        self.logger = LoggerManager().get_logger(self.__class__.__name__)
        self.app_name = app_name
        self.master = master
        self.executor_memory = executor_memory
        self.driver_memory = driver_memory
        self.executor_cores = executor_cores
        self.spark = None
        self._init_spark_session()

    def _init_spark_session(self) -> None:
        Load cryptocurrency data into Spark DataFrame

        Args:
            file_path: Path to data file
            format: File format (csv, json, parquet)

        Returns:
            Spark DataFrame with crypto data
        Create advanced features using PySpark for big data processing

        Args:
            df: Input DataFrame
            symbol_col: Symbol column name
            price_col: Price column name
            volume_col: Volume column name
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with advanced features
        try:
            window = Window.partitionBy(symbol_col).orderBy(timestamp_col)
            window_period = window.rowsBetween(-period + 1, 0)

            df_rsi = df.withColumn(
                "price_diff", col(price_col) - lag(price_col, 1).over(window)
            ).withColumn(
                "gain", when(col("price_diff") > 0, col("price_diff")).otherwise(0)
            ).withColumn(
                "loss", when(col("price_diff") < 0, -col("price_diff")).otherwise(0)
            )

            df_rsi = df_rsi.withColumn(
                "avg_gain", avg("gain").over(window_period)
            ).withColumn(
                "avg_loss", avg("loss").over(window_period)
            )

            rsi = 100 - (100 / (1 + col("avg_gain") / col("avg_loss")))

            return rsi

        except Exception as e:
            self.logger.error(f"Failed to calculate RSI: {e}")
            return lit(50.0)  # Return neutral RSI on error

    def process_large_dataset(self, df: DataFrame,
                             batch_size: int = 10000) -> DataFrame:
        try:
            num_partitions = max(1, df.count() // batch_size)
            df_partitioned = df.repartition(num_partitions)

            df_partitioned.cache()

            self.logger.info(f"Dataset partitioned into {num_partitions} partitions")
            return df_partitioned

        except Exception as e:
            self.logger.error(f"Failed to process large dataset: {e}")
            raise

    def aggregate_market_data(self, df: DataFrame,
                             time_window: str = "1 hour",
                             symbol_col: str = "symbol",
                             timestamp_col: str = "timestamp") -> DataFrame:
        try:
            df_windowed = df.withColumn(
                "window", window(col(timestamp_col), time_window)
            )

            df_agg = df_windowed.groupBy(symbol_col, "window").agg(
                spark_sum("volume").alias("total_volume"),
                avg("close").alias("avg_price"),
                spark_min("low").alias("min_price"),
                spark_max("high").alias("max_price"),
                stddev("close").alias("price_volatility"),
                count("*").alias("trade_count")
            ).withColumn(
                "window_start", col("window.start")
            ).withColumn(
                "window_end", col("window.end")
            ).drop("window")

            self.logger.info(f"Market data aggregated by {time_window}")
            return df_agg

        except Exception as e:
            self.logger.error(f"Failed to aggregate market data: {e}")
            raise

    def train_spark_ml_model(self, df: DataFrame,
                            feature_cols: List[str],
                            target_col: str = "target",
                            model_type: str = "rf") -> Any:
        try:
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol="features"
            )

            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )

            if model_type.lower() == "rf":
                model = RandomForestRegressor(
                    featuresCol="scaled_features",
                    labelCol=target_col,
                    numTrees=100,
                    maxDepth=10,
                    seed=42
                )
            elif model_type.lower() == "gbt":
                model = GBTRegressor(
                    featuresCol="scaled_features",
                    labelCol=target_col,
                    maxIter=100,
                    maxDepth=8,
                    seed=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            pipeline = Pipeline(stages=[assembler, scaler, model])

            model_pipeline = pipeline.fit(df)

            self.logger.info(f"Spark ML model ({model_type}) trained successfully")
            return model_pipeline

        except Exception as e:
            self.logger.error(f"Failed to train Spark ML model: {e}")
            raise

    def save_processed_data(self, df: DataFrame, output_path: str,
                           format: str = "parquet",
                           mode: str = "overwrite") -> None:
        try:
            if format.lower() == "parquet":
                df.write.mode(mode).parquet(output_path)
            elif format.lower() == "csv":
                df.write.mode(mode).option("header", "true").csv(output_path)
            elif format.lower() == "json":
                df.write.mode(mode).json(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Data saved to {output_path} in {format} format")

        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
            raise

    def get_data_quality_report(self, df: DataFrame) -> Dict[str, Any]:
        try:
            total_rows = df.count()
            total_cols = len(df.columns)

            null_counts = {}
            for col_name in df.columns:
                null_count = df.filter(col(col_name).isNull()).count()
                null_counts[col_name] = {
                    "null_count": null_count,
                    "null_percentage": (null_count / total_rows) * 100 if total_rows > 0 else 0
                }

            numeric_stats = {}
            for col_name, col_type in df.dtypes:
                if col_type in ["int", "double", "float", "bigint"]:
                    stats = df.select(col_name).describe().collect()
                    numeric_stats[col_name] = {
                        "count": stats[0][1],
                        "mean": float(stats[1][1]) if stats[1][1] else None,
                        "stddev": float(stats[2][1]) if stats[2][1] else None,
                        "min": float(stats[3][1]) if stats[3][1] else None,
                        "max": float(stats[4][1]) if stats[4][1] else None
                    }

            quality_report = {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "null_analysis": null_counts,
                "numeric_statistics": numeric_stats,
                "data_types": dict(df.dtypes)
            }

            self.logger.info("Data quality report generated")
            return quality_report

        except Exception as e:
            self.logger.error(f"Failed to generate data quality report: {e}")
            raise

    def convert_to_pandas(self, df: DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        try:
            if limit:
                df = df.limit(limit)

            pandas_df = df.toPandas()
            self.logger.info(f"Converted Spark DataFrame to Pandas: {pandas_df.shape}")
            return pandas_df

        except Exception as e:
            self.logger.error(f"Failed to convert to Pandas: {e}")
            raise

    def stop_spark_session(self) -> None:
        self.stop_spark_session()


class PySparkDataProcessor:

    def __init__(self, spark_manager: PySparkManager):
        self.spark_manager = spark_manager
        self.logger = LoggerManager().get_logger(self.__class__.__name__)

    def process_crypto_pipeline(self,
                              input_path: str,
                              output_path: str,
                              symbols: List[str],
                              start_date: str,
                              end_date: str) -> Dict[str, Any]:
        try:
            self.logger.info("Starting crypto data processing pipeline")

            df_raw = self.spark_manager.load_crypto_data(input_path)

            df_filtered = df_raw.filter(
                (col("symbol").isin(symbols)) &
                (col("timestamp") >= start_date) &
                (col("timestamp") <= end_date)
            )

            df_features = self.spark_manager.create_advanced_features(df_filtered)

            df_clean = df_features.dropna()

            df_processed = self.spark_manager.process_large_dataset(df_clean)

            quality_report = self.spark_manager.get_data_quality_report(df_processed)

            self.spark_manager.save_processed_data(df_processed, output_path)

            summary = {
                "input_records": df_raw.count(),
                "filtered_records": df_filtered.count(),
                "processed_records": df_processed.count(),
                "symbols_processed": symbols,
                "date_range": f"{start_date} to {end_date}",
                "quality_report": quality_report,
                "output_path": output_path
            }

            self.logger.info("Crypto data processing pipeline completed successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Crypto processing pipeline failed: {e}")
            raise


if __name__ == "__main__":
    spark_manager = PySparkManager(
        app_name="CryptoForecastingApp",
        master="local[2]",
        executor_memory="2g",
        driver_memory="1g"
    )

    try:
        from datetime import datetime, timedelta
        import random

        dates = [datetime.now() - timedelta(days=x) for x in range(100)]
        symbols = ["BTC", "ETH", "ADA", "DOT"]

        sample_data = []
        for symbol in symbols:
            base_price = random.uniform(100, 50000)
            for date in dates:
                price = base_price * (1 + random.uniform(-0.05, 0.05))
                sample_data.append({
                    "symbol": symbol,
                    "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": price * random.uniform(0.98, 1.02),
                    "high": price * random.uniform(1.01, 1.05),
                    "low": price * random.uniform(0.95, 0.99),
                    "close": price,
                    "volume": random.uniform(1000000, 10000000)
                })

        df = spark_manager.spark.createDataFrame(sample_data)

        df_features = spark_manager.create_advanced_features(df)

        print("\nSample data with features:")
        df_features.select("symbol", "timestamp", "close", "ma_7d", "volatility_7d", "rsi_14").show(10)

        quality_report = spark_manager.get_data_quality_report(df_features)
        print(f"\nData quality report: {quality_report}")

        print("\nPySpark integration test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")

    finally:
        spark_manager.stop_spark_session()
