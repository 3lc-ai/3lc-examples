# Tools

Train classifier from BB table (select model, bb crop augment parameters (GOOD defaults!), epoch to train etc)

Apply classifier predictions and embeddings to table (replace with existing if done before), optionally also apply to Run and Table (then you specify run only )- ability to save Packmap/Umap transform, so can be used in subsequent runs, so must take param in to load transform as well instead of calculating it.

Populate a table with GOOD image statistics (like Encord)

Export table to Coco/Yolo

Create table from Coco/Yolo



## Planning

+ Each py file in tools contains one "main" exportable function that is the entry point for the tool
+ A tool does exactly one thing (but is highly configurable)
+ Can be invoked as a script with arguments
+ Or can be imported and used as a function (same arguments if possible)