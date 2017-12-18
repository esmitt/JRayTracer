// Camera class
// this defines a perspective camera
// camera is defined by it eye position (camera center), at (target) point, and up vector
// field of view in y, and image aspect ratio. 


import javax.vecmath.*;

public class Camera {
	
	private Vector3f eye, up, at;
	private float fovy, aspect_ratio;

	// ****
	private Vector3f center, corner, across;

	public Camera(Vector3f _eye, Vector3f _at, Vector3f _up,
					float _fovy, float _ratio)	{

		// we will use a default camera
		eye = new Vector3f(_eye);
		up = new Vector3f(_up);
		at = new Vector3f(_at);

		fovy = _fovy;
		aspect_ratio = _ratio;

		/* Code for initializing camera
		 * Compute the four corner points of the camera's image plane */
		float dist = 1.f;
		float top = dist * (float)Math.tan(fovy * Math.PI / 360.f);
		float bottom = -top;
		float right = aspect_ratio * top;
		float left = -right;
		Vector3f gaze = new Vector3f();
		gaze.sub(at, eye);

		center = eye;
		Vector3f W = gaze;
		W.negate();
		W.normalize();
		Vector3f V = up;
		Vector3f U = new Vector3f();
		U.cross(V, W);
		U.normalize();
		V.cross(W, U);

		corner = new Vector3f();
		corner.scaleAdd(left, U, center);
		corner.scaleAdd(bottom, V, corner);
		corner.scaleAdd(-dist, W, corner);

		across = new Vector3f(U);
		across.scale(right-left);
		
		up = new Vector3f(V);
		up.scale(top-bottom);
	}

	Ray getCameraRay(float x, float y)
	{
		/* getCameraRay function
		 * (x,y) is a normalized image coordinate, where
		 * both of them vary between [0,1] */

		Vector3f direction = new Vector3f();
		direction.scaleAdd(x, across, corner);
		direction.scaleAdd(y, up, direction);
		direction.sub(center);
		direction.normalize();
		return new Ray(center, direction);
	}
}
