import QtQuick 2.0

Item {
	id: root
	transform: Rotation { origin.x: root.width/2; origin.y: root.height/2; axis { x: 1; y: 0; z: 0 } angle: 180 }

	Rectangle {
		y: 1000; x: 1683
		width: 260; height: 80
		color: "black"
	}
	Image {
		y: 1016; x: 1705
		width: 200; height: 64
		source: "RWLogo.png"
	}
	Rectangle {
		y: 0; x: 0
		width: 130; height: 70
		color: "black"
		visible: false
		objectName:"RecMain"
	}
	Text {
		y: 25; x: 70
     		text: "Rec"
     		font.pointSize: 18
     		color: "red"
		objectName:"TextMain"
		visible: false
  	}	
	Rectangle {
		y: 440; x: 780
		width: 520; height:100
		color: "black"
	}
	Text {
		y: 460; x: 800
     		text: "Pre-Trigger Content"
     		font.pointSize: 40
     		color: "white"
		objectName:"Main"
  	}	
	AnimatedImage { 
		visible: false
		objectName:"GifMain"
		id: animation; source: "rec.gif" 
		width: 70; height: 70
	}
}


