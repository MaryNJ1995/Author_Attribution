#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=no-member
# pylint: disable-msg=import-error
# pylint: disable-msg=not-callable
"""
app.py
"""
import json
import pika
from model_inference import TestModel

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "11/14/2021"


def run():
    """
    run a specific queque in backend
    """
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", port=5672))
    channel = connection.channel()
    channel.queue_declare(queue="authorAttributionOnNews")

    def on_request(char, method, props, body):
        try:
            data = body.decode("utf-8")
            # BackEnd Data
            test_model = TestModel()
            input_json = json.loads(data)
            if "data" in input_json and input_json['data'] == "checkHealth":
                char.basic_publish(exchange='',
                                   routing_key=props.reply_to,
                                   properties=pika.BasicProperties(
                                       correlation_id=props.correlation_id),
                                   body=json.dumps({"result": "ok"}))
                char.basic_ack(delivery_tag=method.delivery_tag)
                return
            # AI output data
            result = test_model.run_flask(input_json)
            char.basic_publish(exchange="",
                               routing_key=props.reply_to,
                               properties=pika.BasicProperties(correlation_id=props.correlation_id),
                               body=json.dumps({"result": result}))
            char.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as error:
            print(error)
            char.basic_publish(exchange="",
                               routing_key=props.reply_to,
                               properties=pika.BasicProperties(
                                   correlation_id=props.correlation_id),
                               body=json.dumps({"result": "error"}))
            char.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="authorAttributionOnNews",
                          on_message_callback=on_request)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()


if __name__ == "__main__":
    run()
