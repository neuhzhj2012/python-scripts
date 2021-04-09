from kafka import KafkaConsumer

if __name__ == '__main__':

    consumer = KafkaConsumer('post-info-cache',
                             bootstrap_servers=[
                                 'log-kafka-1.soulapp.cn:9092',
                                 'log-kafka-2.soulapp.cn:9092',
                                 'log-kafka-3.soulapp.cn:9092'
                             ],
                             # auto_offset_reset='',
                             auto_offset_reset='latest',  # 消费kafka中最近的数据，如果设置为earliest则消费最早的数据，不管这些数据是否消费
                             # auto_offset_reset='earliest',  # 消费kafka中最近的数据，如果设置为earliest则消费最早的数据，不管这些数据是否消费
                             enable_auto_commit=True,  # 自动提交消费者的offset
                             auto_commit_interval_ms=3000,  ## 自动提交消费者offset的时间间隔
                             group_id='test',
                             consumer_timeout_ms=10000,  # 如果10秒内kafka中没有可供消费的数据，自动退出
                             client_id='consumer-python3'
                             )

    for msg in consumer:
        # print (msg.value)
        msg = eval(msg.value)
        print (msg.keys())
        score = msg.get("score")
        img_url = msg.get('imageUrl')
        print (str(score)+"\t"+ img_url)

