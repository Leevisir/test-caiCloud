import sys
from server import app
import json
import unittest

class TestServer(unittest.TestCase):

    def test_fail_without_data(self):
        tester = app.test_client(self)
        response = tester.post('/', content_type='html/text')
        self.assertEqual(response.status_code, 500)
        self.assertTrue(b'Lacking data!' in response.data)

    def test_predict_responds_with_data(self):
        data = dict()
        with open('test/test_data.csv', 'rb') as f:
            data['file'] = (f, f.name)
            tester = app.test_client(self)
            response = tester.post('/', content_type='multipart/form-data', data=data)
            self.assertEqual(response.status_code, 200)

    def test_predict_results(self):
        data = dict()
        with open('test/test_data.csv', 'rb') as f:
            data['file'] = (f, f.name)
            tester = app.test_client(self)
            response = tester.post('/', content_type='multipart/form-data', data=data)
            rd = json.loads(response.data)
            historical = rd['historical']
            fitted = rd['fitted']
            predicted = rd['predicted']
            self.assertEqual(len(predicted['times']),len(predicted['traffic']))
            self.assertEqual(len(fitted['times']),len(fitted['traffic']))
            self.assertEqual(len(historical['times']),len(historical['traffic']))
            self.assertEqual(len(historical['times']),len(fitted['times']))
            self.assertGreater(len(predicted['times']),0)

