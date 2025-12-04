import 'package:flutter/material.dart';

class Tabbar extends StatelessWidget {
  const Tabbar({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
        length: 3,
        child: Scaffold(
          appBar: AppBar(
            title:Text("Tab Bar Example"),
            bottom: const TabBar(
                tabs:[
                  Tab(icon: Icon(Icons.home),text: 'Home',),
                  Tab(icon: Icon(Icons.search), text:"Search"),
                  Tab(icon: Icon(Icons.person),text:'Profile'),
                ],
            ),
          ),
          body: const TabBarView(
            children: [
              HomeScreen(),
              SearchScreen(),
              ProfileScreen(),
            ],
          ),
        ),
    );
  }
}
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text("Home Screen"),
    );
  }
}
class SearchScreen extends StatelessWidget {
  const SearchScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text("Search Screen"),
    );
  }
}
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text("Profile Screen"),
    );
  }
}
