import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import java.time.LocalTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/**
  * Created by ALINA on 24.02.2018.
  */
object CrimeRFClassifier {

  def main(args: Array[String]): Unit = {

    val inputFile = args(0);
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    //Read file to DF
    val crimes = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv(inputFile)

    crimes.show(100)
    crimes.printSchema()

    import sparkSession.implicits._;

    val dayOrNight = udf {
      (h: Int) =>
        if (h > 5 && h < 18) {
          "Day"
        } else {
          "Night"
        }
    }

    val weekend = udf {
      (day: String) =>
        if (day == "Sunday" || day == "Saturday") {
          "Weekend"
        } else {
          "NotWeekend"
        }
    }

    val df = crimes
      .withColumn("HourOfDay", hour(col("Dates")))
      .withColumn("Month", month(col("Dates")))
      .withColumn("Year", year(col("Dates")))
      .withColumn("HourOfDay", hour(col("Dates")))

    val df1 = df
      .withColumn("DayOrNight", dayOrNight(col("HourOfDay")))
      .withColumn("Weekend", weekend(col("DayOfWeek")))

    var categoryIndex = new StringIndexer().setInputCol("Category").setOutputCol("CategoryIndex")
    var dayIndex = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekIndex")
    var districtIndex = new StringIndexer().setInputCol("PdDistrict").setOutputCol("PdDistrictIndex")
    // var addressIndex = new StringIndexer().setInputCol("Address").setOutputCol("AddressIndex")
    var dayNightIndex = new StringIndexer().setInputCol("DayOrNight").setOutputCol("DayOrNightsIndex")

    val assembler = new VectorAssembler().setInputCols(Array(
      "DayOfWeekIndex", "PdDistrictIndex", "HourOfDay", "Month"))
      .setOutputCol("indexedFeatures")

    val Array(training, test) = df1.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier()
      .setLabelCol("CategoryIndex")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(10)
      .setMaxBins(100)

    val pipeline = new Pipeline()
      .setStages(Array(categoryIndex, dayIndex, districtIndex, assembler, rf))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("CategoryIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val model = cv.fit(training)

    val predictions = model.transform(test)
    predictions.show()

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
  }
}
