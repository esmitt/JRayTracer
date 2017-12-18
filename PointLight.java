// PointLight class

import javax.vecmath.*;

public class PointLight extends Light
{
	PointLight(Vector3f pos, Color3f intens)
	{
		position = new Vector3f(pos);
		intensity = new Color3f(intens);
	}

	public Color3f getLight(Vector3f p, Vector3f lightPos, Vector3f lightDir)
	{
		// lightPos
		lightPos.set(position);
		// lightDir
		lightDir.set(lightPos);
		lightDir.sub(p);
		// return lightIntensity
		float r = lightDir.length();
		Color3f lightIntens = new Color3f(intensity);
		lightIntens.scale(1.f / (r*r));			// distance quadratic fall-off
		lightDir.normalize();					// lighting direction should be normalized
		return lightIntens;
	}

	private Vector3f position;	// position of the point light
}


