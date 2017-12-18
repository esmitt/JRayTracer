

public abstract class Shape {

	// each shape must have a hit function
	// which takes an incoming ray, the tmin and tmax
	// and returns a hit structure
	abstract public HitRecord hit(Ray ray, float tmin, float tmax);

	// each shape also has a shadowHit function
	// which is almost the same with hit, but does not return the hit structure
	// by default, we can implement shadowHit by calling hit, and discarding the hit structure
	public boolean shadowHit(Ray ray, float tmin, float tmax)
	{
		return (hit(ray, tmin, tmax) == null) ? false : true;
	}

	protected Material material;	// material of each shape
}
