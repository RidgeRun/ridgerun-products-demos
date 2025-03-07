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
		y: 0; x: 1573
		width: 347; height: 70
		color: "black"
	}
	Text {
		y: 16; x: 1590
     		text: "_"
     		font.pointSize: 20
     		color: "white"
     		objectName: "labelMain"
  	}	
	Rectangle {
		y: 0; x: 0
		width: 130; height: 70
		color: "black"
	}
	Text {
		y: 25; x: 70
     		text: "Rec"
     		font.pointSize: 20
     		color: "red"
  	}	
	AnimatedImage { 
		id: animation; source: "rec.gif" 
		width: 70; height: 70
	}
	Rectangle {
		y: 510; x: 960
		width: 5; height: 65
		color: "red"
	}
	Rectangle {
		y: 540; x: 930
		width: 65; height: 5
		color: "red"
	}
}


