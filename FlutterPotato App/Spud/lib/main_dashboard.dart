import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'prediction.dart'; // Import the Prediction class

class MainDashboard extends StatefulWidget {
  @override
  _MainDashboardState createState() => _MainDashboardState();
}

class _MainDashboardState extends State<MainDashboard> {
  File? _image;
  Prediction? _prediction; // Use the Prediction class instead of String
  String? _token;

  final picker = ImagePicker();
  final dio = Dio();

  Future getImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      } else {
        print('No image selected.');
      }
    });
  }

 Future sendImage() async {
  if (_image == null) return;

  FormData formData = FormData.fromMap({
    'file': await MultipartFile.fromFile(
      _image!.path,
      filename: 'image.jpg',
    ),
    'token': _token ?? '', // Assign empty string if token is null
  });

  try {
    Response response = await dio.post(
       // 'http://10.0.2.2:8000/predict',//local testing
      'https://fastapi-application-spud-ml-app.onrender.com/', //render.com server
      data: formData,
      options: Options(
        contentType: 'multipart/form-data',
      ),
    );

    setState(() {
      _prediction = Prediction.fromJson(response.data);
    });
  } catch (error) {
    print('Error sending image: $error');
  }
}


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Potato Leaf Health Checker',
          style: TextStyle(
            color: Colors.white, // Text color
            fontWeight: FontWeight.bold, // Text font weight
          ),
        ),
        backgroundColor: Colors.green[700], // Background color
        elevation: 0, // Elevation (shadow)
        centerTitle: true, // Center align title
      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/images/logo10.png'),
            fit: BoxFit.cover,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(25),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      spreadRadius: 2,
                      blurRadius: 5,
                      offset: Offset(0, 3),
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(15),
                  child: _image == null
                      ? Center(child: Text('No image selected.'))
                      : Image.file(
                          _image!,
                          width: 300,
                          height: 300,
                          fit: BoxFit.cover,
                        ),
                ),
              ),
              SizedBox(height: 70),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: getImage,
                    style: ButtonStyle(
                      backgroundColor:
                          MaterialStateProperty.all<Color>(Colors.lightGreen),
                      elevation: MaterialStateProperty.all<double>(5),
                      shadowColor:
                          MaterialStateProperty.all<Color>(Colors.grey[300]!),
                    ),
                    child: const Text(
                      ' Select Image ',
                      style: TextStyle(
                        fontSize: 15,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  SizedBox(width: 20),
                  ElevatedButton(
                    onPressed: sendImage,
                    style: ButtonStyle(
                      backgroundColor:
                          MaterialStateProperty.all<Color>(Colors.lightGreen),
                      elevation: MaterialStateProperty.all<double>(5),
                      shadowColor:
                          MaterialStateProperty.all<Color>(Colors.grey[300]!),
                    ),
                    child: const Text(
                      'Check Health',
                      style: TextStyle(
                        fontSize: 15,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 70),
              _prediction == null
                  ? Container()
                  : Container(
                      padding: EdgeInsets.all(10),
                      margin: EdgeInsets.symmetric(horizontal: 20),
                      decoration: BoxDecoration(
                        color: Colors.lightGreen,
                        borderRadius: BorderRadius.circular(25),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            ' Health: ${_prediction!.className} ',
                            style: TextStyle(fontSize: 16, color: Colors.white),
                          ),
                          SizedBox(height: 5),
                          Text(
                            ' Confidence: ${_prediction!.confidence} ',
                            style: TextStyle(fontSize: 16, color: Colors.white),
                          ),
                        ],
                      ),
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
