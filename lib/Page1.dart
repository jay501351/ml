import 'package:exam_practical/Page2.dart';
import 'package:flutter/material.dart';

class Page1 extends StatelessWidget {
  Page1({super.key});

  final TextEditingController textEditingController1 = TextEditingController();
  final TextEditingController textEditingController2 = TextEditingController();


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Page 1"),
        titleTextStyle: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        centerTitle: true,
        backgroundColor: Colors.blueAccent,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          TextField(
            controller: textEditingController1,
            decoration: InputDecoration(
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(5)
              ),
              enabledBorder:OutlineInputBorder(
                borderRadius: BorderRadius.circular(5),
                borderSide:BorderSide(color: Colors.black)
              ),
              hintText:'Enter First Name',
              prefixIcon: Icon(Icons.person)
            ),
          ),
          SizedBox(height: 16.0,),
          TextField(
            controller: textEditingController2,
            decoration: InputDecoration(
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(5)
                ),
                enabledBorder:OutlineInputBorder(
                    borderRadius: BorderRadius.circular(5),
                    borderSide:BorderSide(color: Colors.black)
                ),
                hintText:'Enter Last Name',
                prefixIcon: Icon(Icons.person_2)
            ),
          ),
          SizedBox(height: 16.0,),
          ElevatedButton(onPressed: (){
            String firstname = textEditingController1.text;
            String lastname = textEditingController2.text;

            String? fullname = concatenation(firstname, lastname,context);

            if(fullname!=null){
              Navigator.push(context,
                  MaterialPageRoute(builder: (context)=> Page2(fullName: fullname)));
            }
          }, child:Text("Submit"),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blueAccent,
              foregroundColor: Colors.white,
              minimumSize: Size(190, 54),
              textStyle: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              )
          ),)
        ],
      ),
    );
  }
  String? concatenation(String firstName, String lastName,BuildContext context){
    if(textEditingController1.text.isEmpty || textEditingController2.text.isEmpty){
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("ENter firstname or last name"))
      );
      return null;
    }
    return "$firstName $lastName";
  }
}