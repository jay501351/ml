import 'package:flutter/material.dart';

class Page2 extends StatelessWidget {
  final String fullName;
  const Page2({super.key,required this.fullName});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
          title: Text("Page 2"),
          titleTextStyle: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
          centerTitle: true,
          backgroundColor: Colors.blueAccent,
      ),
      body: Center(
        child:Text('Welcome, $fullName')
      ),
    );
  }
}
