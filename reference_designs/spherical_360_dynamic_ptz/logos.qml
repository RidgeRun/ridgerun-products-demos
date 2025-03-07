import QtQuick 2.0

Item {
	id: root
	transform: Rotation { origin.x: root.width/2; origin.y: root.height/2; axis { x: 1; y: 0; z: 0 } angle: 180 }

	Rectangle {
		y: 1000; x: 840
		width: 260; height: 80
		color: "black"
	}
	Image {
		y: 1016; x: 875
		width: 200; height: 64
		source: "RWLogo.png"
	}
}


