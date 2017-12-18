// Plane class
// defines an infinite plane

import javax.vecmath.*;

public class Plane extends Shape {
	private Vector3f P0;	// a point on the plane
	private Vector3f N ; // plane normal

	public Plane() {
	}
	public Plane(Vector3f p0, Vector3f n, Material mat) {
		P0 = new Vector3f(p0);
		N = new Vector3f(n);
		material = mat;
	}
	public HitRecord hit(Ray ray, float tmin, float tmax) {

		/* compute ray-plane intersection */
		Vector3f temp = new Vector3f(P0);
		temp.sub(ray.getOrigin());
		float denom = ray.getDirection().dot(N);
		if (denom == 0.f)
			return null;
		float t = temp.dot(N) / denom;
		/* if t out of range, return null */
		if (t < tmin || t > tmax)	return null;
		/* construct hit record */
		HitRecord rec = new HitRecord();
		rec.pos = ray.pointAt(t);		// position of hit point
		rec.t = t;						// parameter t (distance along the ray)
		rec.material = material;		// material
		rec.normal = N;					// normal at the hit point
		rec.normal.normalize();			// normal should be normalized
		
		return rec;
	}

}
