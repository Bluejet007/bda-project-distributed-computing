import pyspark.sql as ps
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import pandas
import matplotlib.pyplot as plt
from google.cloud import storage
inp_file_path: str = 'gs://my-bucket-spark-r/2019-Oct.csv'


def save_plot(plot_name):
    local_path = f"/tmp/{plot_name}.png"
    gcs_path = f"output/{plot_name}.png"

    plt.savefig(local_path, bbox_inches="tight")
    plt.close()

    client = storage.Client()
    bucket = client.bucket("my-bucket-spark-r")
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

    print(f"Uploaded gs://my-bucket-spark-r/{gcs_path}")


spark: ps.SparkSession = ps.SparkSession.builder.master("yarn").appName("test").getOrCreate()
raw_data: ps.DataFrame = spark.read.csv(inp_file_path, header=True, inferSchema=True)

raw_data.show(5)
raw_data.summary()


str_ind = StringIndexer(inputCols=['event_type', 'category_code'], outputCols=['event_ind', 'category_ind'], handleInvalid='keep')
str_ind = str_ind.fit(raw_data)
eve_dict = {label: ind for ind, label in enumerate(str_ind.labelsArray[0])}
cat_dict = {label: ind for ind, label in enumerate(str_ind.labelsArray[1])}

dataset = str_ind.transform(raw_data)
dataset = dataset.drop(dataset.category_id, dataset.event_type, dataset.category_code). \
    withColumnsRenamed({'event_ind': 'event_type', 'category_ind': 'category_code'}). \
    withColumn('event_time', F.to_date(dataset.event_time))


print(eve_dict)
print(cat_dict)
dataset.show()
print(dataset.summary())


visits = dataset.groupBy(F.to_date(dataset.event_time).alias("event_time")). \
    agg(F.count_distinct("user_session").alias("number_of_daily_visits"),
        F.count_distinct("user_id").alias("number_of_daily_visitors"))

sales = dataset.where(dataset.event_type == eve_dict['purchase']). \
    withColumn("event_time", F.to_date(dataset.event_time)). \
    groupBy("event_time"). \
    agg(F.count("event_type").alias("number_of_daily_sales"),
        F.sum("price").alias("total_daily_sales"))

daily = visits.join(sales, on=["event_time"], how="left"). \
    withColumn("conversion_rate", F.col('number_of_daily_sales') / F.col('number_of_daily_visits'))

daily.show()


daily_pd = daily.withColumn('day', F.date_format('event_time', 'E')). \
    select('day', 'number_of_daily_visits'). \
    toPandas()
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data_for_plot = [daily_pd[daily_pd['day'] == d]['number_of_daily_visits'] for d in day_order]

plt.boxplot(data_for_plot, labels=day_order)
plt.title('Number of Visits by Days')
plt.ylabel('Number of Visits')
plt.xlabel('Days')
save_plot('Number of Visits by Days')


daily_pd = daily.withColumn('day', F.date_format('event_time', 'E')). \
select('event_time', 'day', 'number_of_daily_visitors'). \
toPandas()

plt.scatter(daily_pd['event_time'], daily_pd['number_of_daily_visitors'])
plt.title('Daily Visitors')
plt.ylabel('Number of Daily Visitors')
plt.xlabel('Dates')
plt.xticks(rotation=45, ha='right')
save_plot('Daily Visitors')

data_for_plot = [daily_pd[daily_pd['day'] == d]['number_of_daily_visitors'] for d in day_order]

plt.boxplot(data_for_plot, labels=day_order)
plt.title('Number of Visitors by Days')
plt.ylabel('Number of Visitors')
plt.xlabel('Days')
save_plot('Number of Visitors by Days')


daily_pd = daily.withColumn('day', F.date_format('event_time', 'E')). \
    select('event_time', 'day', 'conversion_rate'). \
    toPandas()

plt.scatter(daily_pd['event_time'], daily_pd['conversion_rate'])
plt.title('Daily Conversion Rate')
plt.ylabel('Conversion Rate')
plt.xlabel('Dates')
plt.xticks(rotation=45, ha='right')
save_plot('Daily Conversion Rate')

data_for_plot = [daily_pd[daily_pd['day'] == d]['conversion_rate'] for d in day_order]

plt.boxplot(data_for_plot, labels=day_order)
plt.title('Conversion Rate by Days')
plt.ylabel('Conversion Rate')
plt.xlabel('Days')
save_plot('Conversion Rate by Days')


customer_table = dataset.where(dataset.event_type == eve_dict['purchase']). \
    groupBy('user_id'). \
    agg(F.count('user_id').alias('number_of_purchases'),
        F.sum('price').alias('total_sales'))
customer_table.show()


customers_who_purchased = customer_table.select(customer_table.user_id). \
    dropDuplicates(). \
    count()

repeat_customers = customer_table.where('number_of_purchases > 1'). \
    select(customer_table.user_id). \
    dropDuplicates(). \
    count()
    
print(f'There are {customers_who_purchased} customers, who purchased in October, out of these {repeat_customers} are repeat customers.\n')

print('Distribution of Customer by Number of Purchases:')
customer_table.groupBy(customer_table.number_of_purchases).count().show(10)


sales_thresholds = customer_table.approxQuantile("total_sales", [0.01, 0.95], 0.0)
customers_filtered = customer_table.filter((customer_table.total_sales >= sales_thresholds[0]) &
                                           (customer_table.total_sales <= sales_thresholds[1]))

plt.hist(customers_filtered.select(customers_filtered.total_sales).toPandas(), 50)
plt.ylabel('Sales amount')
plt.xlabel('Number of customers')
save_plot('Sales histogram')


top10p_sales = customer_table.approxQuantile("total_sales", [0.9], 0.0)[0]
top10p_customers = customer_table.filter(F.col("total_sales") >= top10p_sales)
regular_customers = customer_table.filter(F.col("total_sales") < top10p_sales)

print(r'Top 10% customers Purchase Amount-Descriptive Statistics')
top10p_customers.select(top10p_customers.total_sales).summary().show()
print(r'Total sales of top 10% customers:')
top10p_customers.agg(F.sum(top10p_customers.total_sales)).show()


print(r'Bottom 90% customers Purchase Amount-Descriptive Statistics')
regular_customers.select(regular_customers.total_sales).summary().show()
print(r'Total sales of bottom 90% customers:')
regular_customers.agg(F.sum(regular_customers.total_sales)).show()


k_mod = Pipeline(stages=[
    VectorAssembler(inputCols=['number_of_purchases', 'total_sales'], outputCol='features'),
    KMeans(k=3, predictionCol='cluster')
])

k_mod = k_mod.fit(regular_customers)
regular_customers = k_mod.transform(regular_customers)


customer_table_joined = customer_table.join(regular_customers.select('user_id', 'cluster'), 'user_id')

customer_table_joined.groupBy(customer_table_joined.cluster).agg(F.sum('total_sales'),
                                                                 F.max('total_sales'),
                                                                 F.min('total_sales'),
                                                                 F.avg('total_sales'),
                                                                 F.stddev('total_sales')).show()

customer_plt = customer_table_joined.select('cluster', 'total_sales'). \
    withColumn('cluster', F.col('cluster') + 1). \
    toPandas()
customer_data = [group["total_sales"].values for _, group in customer_plt.groupby("cluster")]

plt.boxplot(customer_data, labels=sorted(customer_plt["cluster"].unique()))
plt.title("Clusters' Total Sales")
plt.xlabel('Clusters')
plt.ylabel('Total Sales')
save_plot("Clusters' Total Sales")


shopper_list = dataset.where(f'event_type == {eve_dict["purchase"]}').select('user_id').distinct()
dataset_shoppers = shopper_list.join(dataset, 'user_id')

def extract_cat(txt: float | int):
    for k, v in zip(cat_dict.keys(), cat_dict.values()):
        if v == int(txt):
            return k.split('.')[0]
extract_cat_udf = F.udf(extract_cat)

def extract_subcat(txt: float | int):
    for k, v in zip(cat_dict.keys(), cat_dict.values()):
        if v == int(txt):
            return k.split('.')[1]
extract_subcat_udf = F.udf(extract_subcat)


dataset_shoppers = dataset_shoppers.withColumn('category_lvl_1', extract_cat_udf(dataset_shoppers.category_code)).where('category_lvl_1 != "null"')

shopper_visit_table: ps.DataFrame = dataset_shoppers.groupBy('user_id', 'category_lvl_1'). \
    count(). \
    withColumnRenamed('count', 'visits')
shopper_visit_freq: ps.DataFrame = dataset_shoppers.groupBy('user_id'). \
    count(). \
    withColumnRenamed('count', 'frequency')

shopper_visit_table = shopper_visit_table.join(shopper_visit_freq, 'user_id', 'left')
shopper_visit_table = shopper_visit_table.withColumn('ratio', F.col('visits') / F.col('frequency'))

shopper_visit_table.show()


focused_shoppers = shopper_visit_table.filter(F.col("ratio") == 1)
diversified_shoppers = shopper_visit_table.filter(F.col("ratio") != 1)


focused_shoppers = focused_shoppers.withColumn(
    "shoppers_interest_groups",
    F.dense_rank().over(ps.Window.orderBy("category_lvl_1"))
)

diversified_pivot = diversified_shoppers. \
    groupBy("user_id"). \
    pivot("category_lvl_1"). \
    agg(F.first("ratio")). \
    fillna(0)

inp_cols = [
    "accessories",
    "apparel",
    "appliances",
    "auto",
    "computers",
    "construction",
    "country_yard",
    "electronics",
    "furniture",
    "kids",
    "medicine",
    "sport",
    "stationery"
]
c_mod = Pipeline(stages=[
    VectorAssembler(outputCol='features', inputCols=inp_cols),
    KMeans(k=4, predictionCol='cluster')
])

if diversified_pivot.count() > 0:
    c_mod = c_mod.fit(diversified_pivot)
    diversified_clustered = c_mod.transform(diversified_pivot)

    cluster_profile = diversified_clustered.groupBy('cluster'). \
        agg(*[F.mean(c).alias(f'mean_{c}') for c in inp_cols]). \
        orderBy('cluster')

    cluster_counts = diversified_clustered.groupBy('cluster'). \
        count(). \
        orderBy('cluster')

    cluster_profile.show()
    cluster_counts.show()


del focused_shoppers
del diversified_shoppers
del diversified_pivot
del shopper_list
del customer_table
del daily
del data_for_plot


dataset = dataset.withColumn('category', extract_cat_udf(dataset.category_code)). \
    withColumn('subcategory', extract_subcat_udf(dataset.category_code))


print(f'Total number of activity: {dataset.count()}')
print(f'Total number of visits: {dataset.select("user_session").distinct().count():.0f}')
print(f'Total number of visitors: {dataset.select("user_id").distinct().count():.0f}')
print(f'Total number of categories: {dataset.select("category").distinct().count():.0f}')
print(f'Total number of subcategories: {dataset.select("subcategory").distinct().count():.0f}')
print(f'Total number of brands: {dataset.select("brand").distinct().count():.0f}')
print(f'Total number of products: {dataset.select("product_id").distinct().count():.0f}')


category_summary_table = dataset.groupBy('category'). \
    agg(
        F.count('category').alias('number_of_views'),
        F.count_distinct('user_id').alias('number_of_users'),
        F.count_distinct('user_session').alias('number_of_sessions')
    )

category_sales_table = dataset.where(f'event_type == {eve_dict["purchase"]}').groupBy('category'). \
    agg(
        F.count('category').alias('number_of_purchase'),
        F.sum('price').alias('amount_of_purchase'),
        F.avg('price').alias('average_purchase_amount'),
        F.count_distinct('user_session').alias('number_of_sessions_with_purchase'),
        F.count_distinct('user_id').alias('number_of_shoppers')
    )

category_summary_table = category_summary_table.join(category_sales_table, 'category', 'left')


category_pd = category_summary_table.select('category', 'number_of_views').toPandas()
category_pd = category_pd.dropna()
category_pd['category'] = category_pd['category'].astype(str)

plt.bar(category_pd['category'], category_pd['number_of_views'])
plt.title('Total number of views by category')
plt.xlabel('Category')
plt.ylabel('Number of views')
plt.xticks(rotation=45, ha='right')
save_plot('Total number of views by category')

users_category_pd = category_summary_table.select('category', 'number_of_users').toPandas()
users_category_pd = users_category_pd.dropna()
users_category_pd['category'] = users_category_pd['category'].astype(str)

plt.bar(users_category_pd['category'], users_category_pd['number_of_users'])
plt.title('Total number of users by category')
plt.xlabel('Category')
plt.ylabel('Number of users')
plt.xticks(rotation=45, ha='right')
save_plot('Total number of users by category')


category_summary_table = category_summary_table.withColumn('conversion_rate', F.col('number_of_purchase') / F.col('number_of_sessions'))
category_summary_plt = category_summary_table.select('category', 'conversion_rate').toPandas()
category_summary_plt = category_summary_plt.dropna()
category_summary_plt['category'] = category_summary_plt['category'].astype(str)

plt.bar(category_summary_plt['category'], category_summary_plt['conversion_rate'])
plt.title('Conversation rates by category')
plt.xlabel('Category')
plt.ylabel('Conversation rates')
plt.xticks(rotation=45, ha='right')
save_plot('Conversation rates by category')
del category_summary_plt


category_turnover_table = category_summary_table.groupBy('category'). \
    agg(F.sum('amount_of_purchase').alias('total_turnover')). \
    withColumn('total_turnover_mil', F.col('total_turnover') / 1000000.0)
category_turnover_plt = category_turnover_table.select('category', 'total_turnover_mil').toPandas()
category_turnover_plt = category_turnover_plt.dropna()
category_turnover_plt['category'] = category_turnover_plt['category'].astype(str)

plt.bar(category_turnover_plt['category'], category_turnover_plt['total_turnover_mil'])
plt.title('Turnover by category')
plt.xlabel('Category')
plt.ylabel('Turnover per million')
plt.xticks(rotation=45, ha='right')
save_plot('Turnover by category')
del category_turnover_plt