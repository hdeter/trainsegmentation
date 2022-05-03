//get image directory
rootDir = getString("Enter the path to your image directory", "/home/user/Documents/images");
rootDir = rootDir + "/";


//set trainng to True
training = true
savedir = "/home/user/Documents/images/label1"
while (training){
	//open image
	waitForUser("Open an image, select regions for ONE label and add them to the ROI manager (Ctrl+T). When finished press OK");
	
	//get savedir
	//get image directory
	savedir = getString("Enter the path to your label directory", savedir);
	savedir = savedir + "/";
	
	while (savedir == rootDir){
		savedir = getString("Mask directory cannot match image directory. Enter the path to you mask directory", "/home/user/Documents/images/label1");
		savedir = savedir + "/"; }
	
	function savemask(savedir){
		//get filename
		title = getTitle();
		roiManager("Add");
		run("From ROI Manager");
		run("Create Mask");
		savename = savedir + title;
		//print(savename)
		saveAs("Tiff", savename);
		close();
		roiManager("Delete");
		run("Remove Overlay"); }
		
	savemask(savedir); 
	
	training = getBoolean("Do you wish to continue labeling images?");}

