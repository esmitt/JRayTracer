// Ray class

import javax.vecmath.*;

public class Ray {

	public Vector3f o;		// ray's origin
	public Vector3f d;		// ray's direction (normalized)

	public Ray() {
		this.o = new Vector3f(0, 0, 0);
		this.d = new Vector3f(0, -1, 0);
	}

	public Ray(Ray ray) {
		this.o = new Vector3f(ray.o);
		this.d = new Vector3f(ray.d);
	}

	public Ray(Vector3f _o, Vector3f _d) {
		this.o = new Vector3f(_o);
		this.d = new Vector3f(_d);
		this.d.normalize();
	}

	// returns the point at t on the ray
	public Vector3f pointAt(float t) {
		
		Vector3f point = new Vector3f();
		point.scaleAdd(t, d, o);
		return point;
	}

	public void setOrigin(Vector3f origin) {
		this.o = new Vector3f(origin);
	}

	public Vector3f getOrigin() { return this.o; }

	public void setDirection(Vector3f direction) {
		this.d = new Vector3f(direction);
		this.d.normalize();
	}
	
	public Vector3f getDirection() { return this.d; }
}
