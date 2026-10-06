"use client";



import { useEffect, useRef, useState } from "react";

import { analyzeFull, whatIf, ApiError } from "@/lib/api/client";

import type { AnalyzeFull } from "@/lib/types/api";



// Read a number from a response object by trying several likely key names.

const pick = (o: any, keys: string[]): number | undefined => {

  for (const k of keys) {

    if (typeof o?.[k] === "number" && Number.isFinite(o[k])) {

      return o[k] as number;

    }

  }

  return undefined;

};



// Percentage display for values stored as probabilities (0–1).

const pct = (x?: number): string =>

  x === undefined ? "N/A" : `${(x * 100).toFixed(1)}%`;



// Compact decimal formatting for readable summaries only.

// Raw JSON below still keeps the original full precision.

const prettyNumber = (

  x: unknown,

  digits = 4

): string => {

  if (typeof x !== "number" || !Number.isFinite(x)) {

    return "N/A";

  }



  return x.toFixed(digits).replace(/\.?0+$/, "");

};



const prettyPercent = (x: unknown, digits = 1): string => {

  if (typeof x !== "number" || !Number.isFinite(x)) {

    return "N/A";

  }



  return `${(x * 100).toFixed(digits)}%`;

};



const formatLabel = (value: unknown): string => {

  if (typeof value !== "string" || !value.trim()) {

    return "N/A";

  }



  return value

    .replace(/\_/g, " ")

    .replace(/\b\w/g, (char) => char.toUpperCase());

};



const initialTestValue = (candidate: any): number => {
  const current = Number(candidate?.current_value);
  if (!Number.isFinite(current)) return 0;

  const low = candidate?.normal_stats
    ? Math.min(candidate.normal_stats.p5, current)
    : current;
  const high = candidate?.failure_stats
    ? Math.max(candidate.failure_stats.p95, current)
    : current;

  if (!Number.isFinite(low) || !Number.isFinite(high) || high === low) {
    return current;
  }

  const midpoint = (low + high) / 2;
  const options = [low, midpoint, high].filter(
    (v): v is number => Number.isFinite(v)
  );

  // Start with a real counterfactual value instead of the current value.
  return options.reduce(
    (best, v) => Math.abs(v - current) > Math.abs(best - current) ? v : best,
    current
  );
};

export default function WhatIfPage() {

  const [id, setId] = useState("100");

  const [base, setBase] = useState<AnalyzeFull | null>(null);

  const [feature, setFeature] = useState("");

  const [value, setValue] = useState<number | null>(null);

  const [res, setRes] = useState<Record<string, any> | null>(null);

  const [err, setErr] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);



  const ctl = useRef<AbortController | null>(null);



  const msg = (e: unknown): string =>

    e instanceof ApiError ? e.message : "Unexpected error.";



  async function load() {

    setErr(null);

    setRes(null);



    try {

      const sampleId = Number(id);



      if (!Number.isFinite(sampleId)) {

        setErr("Please enter a valid sample ID.");

        return;

      }



      const a = await analyzeFull(sampleId, 5);



      setBase(a);



      const c = a.root_cause_candidates?.[0];



      if (c) {
        setFeature(c.feature);
        setValue(initialTestValue(c));
      } else {

        setFeature("");

        setValue(null);

      }

    } catch (e) {

      setErr(msg(e));

    }

  }



  const cand = base?.root_cause_candidates?.find(

    (c) => c.feature === feature

  );



  const lo = cand?.normal_stats

    ? Math.min(cand.normal_stats.p5, cand.current_value)

    : cand?.current_value ?? 0;



  const hi = cand?.failure_stats

    ? Math.max(cand.failure_stats.p95, cand.current_value)

    : cand?.current_value ?? 1;



  const range = Math.max(hi - lo, 0.000001);

  const step = range / 200;



  useEffect(() => {

    if (!base || !feature || value === null) return;



    const sampleId = base.sample?.sample_id;



    if (sampleId === null || sampleId === undefined) return;



    const testValue = value;



    const timer = setTimeout(async () => {

      ctl.current?.abort();



      const controller = new AbortController();

      ctl.current = controller;



      setBusy(true);

      setErr(null);



      try {

        const result = await whatIf(

          sampleId,

          feature,

          testValue,

          controller.signal

        );



        setRes(result);

      } catch (e: unknown) {

        if (

          e &&

          typeof e === "object" &&

          "name" in e &&

          (e as { name?: string }).name === "AbortError"

        ) {

          return;

        }



        setErr(msg(e));

      } finally {

        setBusy(false);

      }

    }, 400);



    return () => {

      clearTimeout(timer);

    };

  }, [base, feature, value]);



const before =

  pick(res, [

    "current_risk",

    "baseline_probability",

    "baseline_risk",

    "original_probability",

  ]) ?? base?.prediction?.failure_probability;



// The backend returns the simulated risk inside curve[].risk.

const simulatedPoint =
  Array.isArray(res?.curve) && typeof value === "number"
    ? res.curve.reduce((closest: any, point: any) => {
        if (typeof point?.risk !== "number" || !Number.isFinite(point.risk)) {
          return closest;
        }
        if (!closest) return point;
        return Math.abs(Number(point.value) - value) <
          Math.abs(Number(closest.value) - value)
          ? point
          : closest;
      }, undefined)
    : undefined;



const after =

  pick(res, [

    "scenario_risk",

    "new_probability",

    "scenario_probability",

    "counterfactual_probability",

    "failure_probability",

  ]) ?? simulatedPoint?.risk;



  // Backend recommendation from /analyze.
  const recommendation = base?.recommendation as
    | Record<string, any>
    | null
    | undefined;

  const hasRecommendation =
    !!recommendation && Object.keys(recommendation).length > 0;

  return (

    <main className="mx-auto max-w-6xl px-6 py-10">

      <h1 className="text-3xl font-bold">What-if simulator</h1>



      <p className="mt-1 text-sm text-zinc-400">

        The selected sensor and curve are evaluated by the FastAPI POST

        /what-if endpoint.

      </p>



      <form

        className="mt-6 flex items-end gap-3"

        onSubmit={(e) => {

          e.preventDefault();

          void load();

        }}

      >

        <label className="text-sm">

          Sample ID

          <input

            value={id}

            onChange={(e) => setId(e.target.value)}

            className="field w-40"

          />

        </label>



        <button type="submit" className="btn-primary">

          Load sample

        </button>

      </form>



      {err && (

        <p role="alert" className="mt-4 text-risk">

          {err}

        </p>

      )}



      {base && cand && value !== null && (

        <div className="mt-6 grid gap-4 md:grid-cols-2">

          {/* Parameter control */}

          <section className="card p-6">

            <label className="eyebrow">

              Parameter



              <select

                value={feature}

                onChange={(e) => {

                  const selectedFeature = e.target.value;

                  setFeature(selectedFeature);



                  const selectedCandidate =

                    base.root_cause_candidates.find(

                      (x) => x.feature === selectedFeature

                    );



                  if (selectedCandidate) {
                    setValue(initialTestValue(selectedCandidate));
                  }

                }}

                className="mt-1 block w-full rounded-md border border-line bg-bg px-3 py-2 text-white"

              >

                {base.root_cause_candidates.map((c) => (

                  <option key={c.feature} value={c.feature}>

                    {c.feature}

                  </option>

                ))}

              </select>

            </label>



            <p className="mt-4 text-sm">

              Current{" "}

              {prettyNumber(cand.current_value)}{" "}

              → test{" "}

              <span className="font-semibold">

                {prettyNumber(value, 3)}

              </span>

            </p>



            <input

              type="range"

              aria-label="Test value"

              className="mt-2 w-full"

              min={lo}

              max={hi}

              step={step}

              value={value}

              onChange={(e) => setValue(Number(e.target.value))}

            />

          </section>



          {/* Live what-if result */}

          <section className="card p-6" aria-live="polite">

            <p className="text-xs text-ok">

              Live · POST /what-if

              {busy && " · updating…"}

            </p>



            <p className="mt-2 text-3xl font-semibold">

              {pct(before)} → {pct(after)}

            </p>



            <p className="eyebrow">

              Change:{" "}

              {before !== undefined &&

              after !== undefined &&

              Number.isFinite(before) &&

              Number.isFinite(after)

                ? `${((after - before) * 100).toFixed(1)} percentage points`

                : "N/A"}

            </p>



            <p className="mt-4 text-xs text-warn">

              Model-based counterfactual. Requires engineering validation

              before any physical process change.

            </p>

          </section>



          {/* Raw what-if response */}

          {res && (

            <details className="card p-6 md:col-span-2">

              <summary className="cursor-pointer text-sm text-zinc-400">

                Raw response

              </summary>



              <pre className="mt-3 max-h-[500px] overflow-auto rounded-md border border-line bg-black/20 p-4 text-xs leading-relaxed">

                {JSON.stringify(res, null, 2)}

              </pre>

            </details>

          )}



          {/* Full backend recommendation JSON */}

          {hasRecommendation && (

            <details className="card p-6 md:col-span-2">

              <summary className="cursor-pointer text-sm text-zinc-400">

                Raw backend recommendation

              </summary>



              <pre className="mt-3 max-h-[500px] overflow-auto rounded-md border border-line bg-black/20 p-4 text-xs leading-relaxed">

                {JSON.stringify(recommendation, null, 2)}

              </pre>

            </details>

          )}

        </div>

      )}

    </main>

  );

}