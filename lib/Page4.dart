import 'package:flutter/material.dart';

class GridScreen extends StatelessWidget {
  const GridScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Generate a list of 20 items
    final List<String> items =
    List.generate(20, (index) => 'Item ${index + 1}');

    return Scaffold(
      appBar: AppBar(
        title: const Text('GridView 3x3 Example'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: GridView.builder(
          itemCount: items.length,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3, // 3 columns
            crossAxisSpacing: 10,
            mainAxisSpacing: 10,
          ),
          itemBuilder: (context, index) {
            return GestureDetector(
              onTap: () {
                // Navigate to detail screen on tap
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => DetailScreen(itemName: items[index]),
                  ),
                );
              },
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.indigo.shade100,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.indigo, width: 1.5),
                ),
                child: Center(
                  child: Text(
                    items[index],
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      color: Colors.indigo,
                    ),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class DetailScreen extends StatelessWidget {
  final String itemName;

  const DetailScreen({super.key, required this.itemName});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(itemName)),
      body: Center(
        child: Text(
          'Details of $itemName',
          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ),
      ),
    );
  }
}