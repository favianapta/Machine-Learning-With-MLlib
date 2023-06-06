# Praktikum Machine Learning With MLlib

<a name="readme-top"></a>

<!-- ml-recommender.scala -->
# ML - Recommender

  ### Kode Sebelum di ubah

   ```css
   import org.apache.spark.ml.recommendation.ALS
 
   case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
   def parseRating(str: String): Rating = {
     val fields = str.split("::")
     assert(fields.size == 4)
     Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
   }

   //Test it
   parseRating("1::1193::5::978300760")

   var raw = sc.textFile("/data/ml-1m/ratings.dat")
   //check one record. it should be res4: Array[String] = Array(1::1193::5::978300760)
   //If this fails the location of file is wrong.
   raw.take(1)

   val ratings = raw.map(parseRating).toDF()
   //check if everything is ok
   ratings.show(5)

   val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

   // Build the recommendation model using ALS on the training data
   //Alternating Least Squares (ALS) matrix factorization.
   val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

   val model = als.fit(training)
   model.save("mymodel")

   //Prepare the recommendations
   val predictions = model.transform(test)
   predictions.map(r => r(2).asInstanceOf[Float] - r(4).asInstanceOf[Float])
   .map(x => x*x)
   .filter(!_.isNaN)
   .reduce(_ + _)

   predictions.take(10)

   predictions.write.format("com.databricks.spark.csv").save("ml-predictions.csv")
   ```
     
  ### Kode Setalah Perubahan di Google Collabs
  
  ```css
  !pip install pyspark
  from pyspark.ml.recommendation import ALS
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import col
  import math

  spark = SparkSession.builder.getOrCreate()
  sc = spark.sparkContext

  class Rating:
      def __init__(self, userId, movieId, rating, timestamp):
          self.userId = userId
          self.movieId = movieId
          self.rating = rating
          self.timestamp = timestamp

  def parseRating(line):
      fields = line.split("::")
      assert len(fields) == 4
      return Rating(int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3]))

  # Test it
  parseRating("1::1193::5::978300760")

  raw = sc.textFile("ratings.dat")
  # check one record. it should be res4: Array[String] = Array(1::1193::5::978300760)
  # If this fails the location of the file is wrong.
  raw.take(1)

  ratings = raw.map(parseRating).toDF()
  # check if everything is ok
  ratings.show(5)

  training, test = ratings.randomSplit([0.8, 0.2])

  # Build the recommendation model using ALS on the training data
  # Alternating Least Squares (ALS) matrix factorization.
  als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")

  model = als.fit(training)
  model.save("mymodel")

  # Prepare the recommendations
  predictions = model.transform(test)
  squared_diff = predictions.select((col("rating").cast("float") - col("prediction").cast("float")).alias("squared_diff")).na.drop()
  squared_diff_squared = squared_diff.select((col("squared_diff") ** 2).alias("squared_diff_squared")).na.drop()
  mse = squared_diff_squared.agg({"squared_diff_squared": "mean"}).collect()[0][0]
  rmse = math.sqrt(mse)

  predictions.take(10)

  predictions.write.format("com.databricks.spark.csv").save("ml-predictions.csv")
  ```
     
  ### HASIL
  
  <img src="images/ML - Recommender.png" width="60%" height="60%">
  
  ### Penjelasan
  
  
  
<!-- mllib_random_forrest.ipynb -->
# MLlib Random Forest

  ### Kode Sebelum di ubah

  ```css
  mylist = [(50, "DataFrame"),(60, "pandas")]
  myschema = ['col1', 'col2']
  ```
     
  ### Kode Setalah Perubahan di Google Collabs
  
  ```css
  mylist = [(50, "DataFrame"),(60, "pandas")]
  myschema = ['col1', 'col2']
  ```
     
  ### HASIL
  BELUM DALAM PROSES
  <img src="images/Kode-1.png" width="60%" height="60%">
    
  + Penjelasan
     -
     
<!-- movie-recommendations.py -->
# Movie Recommendations

  ### Kode Sebelum di ubah

  ```css
  mylist = [(50, "DataFrame"),(60, "pandas")]
  myschema = ['col1', 'col2']
  ```
     
  ### Kode Setalah Perubahan di Google Collabs
  
  ```css
  mylist = [(50, "DataFrame"),(60, "pandas")]
  myschema = ['col1', 'col2']
  ```
    
  ### HASIL
  
  BELUM DALAM PROSES
  
  <img src="images/Kode-1.png" width="60%" height="60%">
    
  + Penjelasan
     -
