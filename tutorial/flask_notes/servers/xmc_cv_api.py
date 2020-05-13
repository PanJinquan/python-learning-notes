import urllib3
import unittest
import json

host="http://127.0.0.1:8888"
# host="http://192.168.4.50:8000"
version_api="/v1/engine/imageprocess/version"
status_api='/v1/engine/imageprocess/status'
analyze_api='/v1/engine/imageprocess/async/analyze'
imageprocesscb_api='/v1/engine/pipelinemanager/imageprocesscb'

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.http = urllib3.PoolManager()

    def test_handle_async_cv_analyze(self):
        '''
        "image_dict": ["2334/1828778894_271415878a_o.jpg", "3149/2355285447_290193393a_o.jpg", "2090/1792526652_8f37410561_o.jpg", "2099/1791684639_044827f860_o.jpg"],
        "storage_url": "https://farm3.staticflickr.com/"
        :return:
        '''
        # json_data = {
        #     "request_id": "value",
        #     "source_id": "value",
        #     "type": "student",
        #     "data": {
        #         "image_dict": ["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
        #         "storage_url": "F:/XMC/dataset/test_image"
        #     },
        #     "callback": imageprocesscb_api
        # }
        json_data = {
            "request_id": "value",
            "source_id": "value",
            "type": "student",
            "data": {
                # "image_dict": ["000000001.jpg" ,"000000002.jpg" ,"000000003.jpg" ,"000000004.jpg" ],
                # "storage_url": "http://192.168.4.50:8000/image"
                "image_dict": ["2334/1828778894_271415878a_o.jpg", "3149/2355285447_290193393a_o.jpg", "2090/1792526652_8f37410561_o.jpg", "2099/1791684639_044827f860_o.jpg"],
                "storage_url": "https://farm3.staticflickr.com/"
            },
            "callback": imageprocesscb_api
        }
        json_data = json.dumps(json_data).encode("utf-8")
        response = self.http.request("POST", host + analyze_api,
                                     body=json_data,
                                     headers={"Content-Type": "application/json"})
        print("analyze_response:{}".format(response.data.decode("utf-8")))

    def test_get_status(self):
        """TODO"""
        response = self.http.request("GET", host+status_api,
                                     headers={"Content-Type": "application/json"})
        print("status:{}".format(response.data.decode("utf-8")))

    # def test_get_version(self):
    #     """TODO"""
    #     response = self.http.request("GET", host + version_api,
    #                                  headers={"Content-Type": "application/json"})
    #     print("version:{}".format(response.data.decode("utf-8")))



    # def test01_post_login(self):
    #     """TODO"""
    #
    # def test02_post_config(self):
    #     """TODO"""
    #     config = {"user_id": "string",
    #               "language": "string",
    #               "bit_rate": "string"}
    #     config = json.dumps(config).encode("utf-8")
    #     response = self.http.request("POST", "http://127.0.0.1/api/v1/sys/status",
    #                                  body=config,
    #                                  headers={"Content-Type": "application/json"})
    #     print(response.data.decode("utf-8"))
    #
    # def test03_post_file(self):
    #     """TODO"""
    #     file_path = "../audios/test.wav"
    #     with open(file_path, "r+", encoding="utf-8") as f:
    #         file = f.read()
    #     response = self.http.request("POST", "http://127.0.0.1/api/v1/sys/status",
    #                                  fields={"filefield": ("test.wav", file, "text/plain")},
    #                                  headers={"Content-Type": "application/json"})
    #     print(response.data.decode("utf-8"))
    #
    # def test05_post_stream(self):
    #     """TODO"""
    #
    # def test06_post_result(self):
    #     """TODO"""
    #     file_path = "../audios/test.wav"
    #     with open(file_path, "r+", encoding="utf-8") as f:
    #         file = f.read()
    #     response = self.http.request("POST", "http://127.0.0.1/api/v1/sys/status",
    #                                  fields={"filefield": ("test.wav", file, "text/plain")},
    #                                  headers={"Content-Type": "application/json"})
    #     print(response.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
