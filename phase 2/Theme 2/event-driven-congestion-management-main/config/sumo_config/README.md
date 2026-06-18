SUMO integration is optional.

Place a SUMO network file here:

- `bangalore.net.xml`

When `sumo`, `netconvert`, and `duarouter` are available on PATH and this network file exists, the Bernoulli diversion stage records SUMO capability and can be extended to run microscopic demand simulations. Without those assets, the pipeline uses the built-in NetworkX Bernoulli pressure simulation.
