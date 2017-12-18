// Sphere class
// defines a Sphere shape

import javax.vecmath.*;

public class Sphere extends Shape {
	private Vector3f center;	// center of sphere
	private float radius;		// radius of sphere

	public Sphere() {
	}

	public Sphere(Vector3f pos, float r, Material mat) {
		center = new Vector3f(pos);
		radius = r;
		material = mat;
	}

	public HitRecord hit(Ray ray, float tmin, float tmax) {
		HitRecord hr;

		float t, t1, t2;

		// Formulas from Foley et al. "Introduction to Computer Graphics" for the
		// algebraic ray-sphere intersection method.
		float a = (ray.d.x * ray.d.x) + (ray.d.y * ray.d.y) + (ray.d.z * ray.d.z);

		float b = (2 * ray.d.x * (ray.o.x - center.x)) + (2 * ray.d.y * (ray.o.y - center.y)) + (2 * ray.d.z * (ray.o.z - center.z));

		float c = (center.x * center.x) + (center.y * center.y) + (center.z * center.z) +
				(ray.o.x * ray.o.x) + (ray.o.y * ray.o.y) + (ray.o.z * ray.o.z) -
				2 * ((center.x * ray.o.x) + (center.y * ray.o.y) + (center.z * ray.o.z)) - (radius * radius);

		// Calculate the determinant.
		float d = (b * b) - (4 * a * c);

		if (d >= 0.0f) {
			// If the determinant is 0 or greater then there is at least one intersection.
			// Then calculate the two possible intersections.
			t1 = (float) ((-b - Math.sqrt(d)) / (2 * a)); 
			t2 = (float) ((-b + Math.sqrt(d)) / (2 * a));

			// Keep the closest intersection.
			t = t1 < t2 ? t1 : t2;

			if (t >= 0.0f) {
				// If the intersection is in front of the ray then it is a valid intersection.

				// Check if the intersection is within bounds.
				if (t < tmin || t > tmax)
					return null;

				// Create the hit record and set it's parameters.
				hr = new HitRecord();

				// T
				hr.t = t;

				// Hit position.
				// Subtract a little from t so that the intersection point is not exactly on
				// the sphere's surface but a little bit above it. This is to avoid self-intersections
				// later when evaluating the shadow rays.
				hr.pos = ray.pointAt(t - 0.000001f);

				// Hit position material
				hr.material = this.material;

				// Calculate the normal vector at the hit point.
				hr.normal = new Vector3f(hr.pos);
				hr.normal.sub(center);
				hr.normal.set(hr.normal.x / radius, hr.normal.y / radius, hr.normal.z / radius);
				hr.normal.normalize();

				return hr;
			}
		}

		// If we reach this point then there is no valid intersection.
		return null;
	}
}
