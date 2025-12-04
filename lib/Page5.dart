import 'dart:async';

import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const SplashScreen(),
      routes: {'/home': (context) => Home()},
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    Timer(Duration(seconds: 3), () {
      Navigator.pushReplacementNamed(context, "/home");
      /*Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) {
        return Home();
      },));*/
    },);

  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Image.asset("assets/images/abc.jpg",width: 150,height: 150,),
          Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              CircularProgressIndicator(
                color: Colors.white,
              ),
              SizedBox(height: 150,)
            ],
          )
        ],
      ),
      backgroundColor: Colors.black,
    );
  }
}

class Home extends StatelessWidget {
  final List<Map<String,String>> list = [
    {
      'title':'image1',
      'image':'https://picsum.photos/400/300/?1'
    },
    {
      'title':'image2',
      'image':'https://picsum.photos/400/300/?2'
    },
    {
      'title':'image3',
      'image':'https://picsum.photos/400/300/?3'

    }
  ];

  Home({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Hero Animation Example"),),
      body: ListView.builder(itemCount: list.length,
        itemBuilder: (context, index) {
          final item = list[index];
          return Card(
            child: ListTile(
              contentPadding: EdgeInsets.symmetric(vertical: 8.0,horizontal: 8.0),
              leading: Hero(tag: item['image']!, child: Image.network(item['image']!)),
              title: Text(item['title']!),
              onTap: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) {
                  return ListDetails(item: item);
                },));
              },
            ),
          );
        },),
    );
  }
}

class ListDetails extends StatelessWidget {
  final Map<String,String> item;
  const ListDetails({super.key,required this.item});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text("List Details"),),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Hero(tag: item['image']!, child: Image.network(item['image']!)),
            SizedBox(height: 10,),
            Text(item['title']!,style: TextStyle(fontSize: 20,fontWeight: FontWeight.bold),),
          ],
        )
    );
  }
}