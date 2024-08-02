from flask import Flask, request
from flask_restx import Api, Resource, fields
from functions import predict_image

app = Flask(__name__)
api = Api(app, version='1.0', title='Proyecto Final API REST para red CNN', description='API para realizar predicciones sobre imágenes meteorológicas')

prediction_model = api.model('Prediction',
                             {'image': fields.String(
                                                        required=True,
                                                        description='Imagen a predecir',
                                                        location='form',
                                                    )})

@api.route('/predict')
class Prediction(Resource):
    @api.expect(prediction_model)
    def post(self):
        image_file = request.files.get('image')

        # Procesamos la imagen y realizamos la prediccion
        prediction = predict_image(image_file)

        return {'prediction': prediction}

if __name__ == '__main__':
    app.run(debug=True)