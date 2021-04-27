import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object TweetProcessing {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: TweetProcessing InputDir OutputDir")
    }

    val inputFilePath = args(0)
    val outputFilePath = args(1)

    // Create a spark session
    val spark = SparkSession
      .builder()
      .appName("Tweet Processing")
      .getOrCreate()

    // Storing the input file in data variable
    var data = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputFilePath)
      .select("tweet_id", "text", "airline_sentiment")
      .toDF("id", "text", "labelS")

    // Removing rows where text is null
    data = data.filter(data.col("text").isNotNull)

    // Splitting input values into training and test data
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2))

    // Configure an ML pipeline, which consists of five stages: tokenizer, stop words remover, hashingTF, indexer and logistic regression.
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W+")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")
    val hashingTF = new HashingTF()
      .setNumFeatures(100)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
    val indexer = new StringIndexer()
      .setInputCol("labelS")
      .setOutputCol("label")
    val lr = new LogisticRegression()
      .setMaxIter(10)
    val pipeline_train = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, indexer, lr))

    // Creating a Parameter Grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(500, 800, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.2, 0.01, 0.05, 0.15))
      .addGrid(lr.fitIntercept)
      .build()

    // Creating a Cross Validator
    val cv = new CrossValidator()
      .setEstimator(pipeline_train)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    // Fitting the training data through Cross Validator which runs it through the pipeline
    val cvModel = cv.fit(training)

    cvModel.bestModel.params.toString

    // Creating the prediction and labels
    val predictionAndLabels = cvModel.transform(test)
      .select("prediction", "label")
      .rdd
      .map(x => (x.getAs[Double](0), x.getAs[Double](1)))

    // Calculating the classification metrics
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Creating a variable to print output to a text file in the end
    var output = ""

    // Confusion matrix
    val confusionMatrix = metrics.confusionMatrix
    output += "Confusion Matrix: \n" + confusionMatrix + "\n"

    // Overall Statistics
    val accuracy = metrics.accuracy
    output += "Summary Statistics \n Accuracy: \t" + accuracy + "\n"

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      output += "Precision(" + l + ") = \t" + metrics.precision(l) + "\n"
    }

    // Recall by label
    labels.foreach { l =>
      output += "Recall(" + l + ") = \t" + metrics.recall(l) + "\n"
    }

    // False positive rate by label
    labels.foreach { l =>
      output += "FPR(" + l + ") = \t" + metrics.falsePositiveRate(l) + "\n"
    }

    // F-measure by label
    labels.foreach { l =>
      output += "F1-Score(" + l + ") = \t" + metrics.fMeasure(l) + "\n"
    }

    // Weighted stats
    output += "\nWeighted Statistics: \n"
    output += "Weighted precision: \t" + metrics.weightedPrecision + "\n"
    output += "Weighted recall: \t" + metrics.weightedRecall + "\n"
    output += "Weighted F1 score: \t" + metrics.weightedFMeasure + "\n"
    output += "Weighted false positive rate: \t" + metrics.weightedFalsePositiveRate + "\n"

    val sc = spark.sparkContext
    val outputRdd: RDD[String] = sc.parallelize(List(output));
    outputRdd.coalesce(1, true).saveAsTextFile(outputFilePath)
  }
}
