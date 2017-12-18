// Light class
// defines the base class Light

import javax.vecmath.*;

public abstract class Light {

	// getLight takes a point in the scene, and returns a point on the light (lightPos),
	// the direction that points from p to lightPos (lightDir)
	// and returns the color (intensity) of the light.
	abstract public Color3f getLight(Vector3f p, Vector3f lightPos, Vector3f lightDir);

	protected Color3f intensity;
}


