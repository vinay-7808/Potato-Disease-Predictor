class Prediction {
  String? className;
  double? confidence;

  Prediction({this.className, this.confidence});

  Prediction.fromJson(Map<String, dynamic> json) {
    className = json['class'];
    confidence = json['confidence']
        ?.toDouble(); // Ensure confidence is converted to double
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['class'] = this.className;
    data['confidence'] = this.confidence;
    return data;
  }
}
