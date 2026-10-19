import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const VERDICT = {
  CLEAN: { label: "CLEAN", badge: "bg-emerald-100 text-emerald-800", banner: "border-emerald-300 bg-emerald-50 text-emerald-800" },
  DAMAGED: { label: "DAMAGED", badge: "bg-red-100 text-red-700", banner: "border-red-300 bg-red-50 text-red-700" },
  UNCERTAIN: { label: "NEEDS REVIEW", badge: "bg-amber-100 text-amber-800", banner: "border-amber-300 bg-amber-50 text-amber-800" },
};
const ACTION_COLOR = {
  "Approve for Policy": "text-emerald-700",
  "Reject / Re-price": "text-red-600",
  "Physical Inspection": "text-amber-700",
  "Re-capture Video": "text-amber-700",
  "Re-capture Photos": "text-amber-700",
};

const errText = (ex) => {
  const d = ex?.response?.data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg).join("; ");
  return ex?.message || "Something went wrong";
};
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const when = (iso) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const isSolid = (t) => t.image_name || t.n_frames >= 3;
const isReal = (t) => isSolid(t) && !t.likely_reflection;
const damageCount = (c) => c.tracks.filter(isReal).length;

function Badge({ c }) {
  if (c.status === "processing") return <span className="rounded px-2 py-1 text-xs font-bold bg-blue-100 text-blue-700 animate-pulse">ANALYSING…</span>;
  if (c.status === "failed") return <span className="rounded px-2 py-1 text-xs font-bold bg-slate-200 text-slate-700">FAILED</span>;
  const v = VERDICT[c.verdict] || VERDICT.UNCERTAIN;
  return <span className={`rounded px-2 py-1 text-xs font-bold ${v.badge}`}>{v.label}</span>;
}

function Stat({ label, value, color }) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm border-t-4" style={{ borderColor: color }}>
      <div className="text-3xl font-bold">{value ?? 0}</div>
      <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}

function Upload({ onCreated }) {
  const ref = useRef();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [drag, setDrag] = useState(false);
  const [vin, setVin] = useState("");
  const vinOk = /^[A-HJ-NPR-Z0-9]{17}$/.test(vin);

  const send = async (fileList) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    if (!vinOk) { setErr("Enter a valid 17-character chassis number first."); return; }
    setBusy(true); setErr("");
    const fd = new FormData();
    fd.append("chassis_no", vin);
    files.forEach((f) => fd.append("files", f));
    try { await axios.post("/api/claims", fd); setVin(""); onCreated(); }
    catch (ex) { setErr(errText(ex)); }
    setBusy(false);
    if (ref.current) ref.current.value = "";
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files); }}
      className={`rounded-xl border-2 border-dashed p-8 text-center bg-white ${drag ? "border-blue-500" : "border-slate-300"}`}>
      <label className="mx-auto mb-4 block max-w-sm text-left text-xs font-semibold text-slate-500">Chassis number (VIN)
        <input value={vin} maxLength={17} placeholder="17 characters, e.g. MA3EWDE1S00123456"
          onChange={(e) => setVin(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, ""))}
          className={`mt-1 w-full rounded-lg border px-3 py-2 font-mono text-sm tracking-wider ${vin && !vinOk ? "border-red-400" : "border-slate-300"}`} />
        <span className="font-normal">{vin.length}/17 · no I, O or Q</span></label>
      <p className="font-semibold">Upload a 360° walk-around video, or photos of every side</p>
      <p className="mt-1 text-sm text-slate-500">Video: walk fully around the car. Photos: at least 4 (front, rear, left, right). Anything that skips a side cannot be certified clean.</p>
      <input ref={ref} type="file" accept="video/*,image/*" multiple hidden onChange={(e) => send(e.target.files)} />
      <button disabled={busy || !vinOk} onClick={() => ref.current.click()}
        className="mt-4 rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50">
        {busy ? "Uploading…" : "Choose video or photos"}</button>
      {err && <p className="mt-3 text-sm text-red-600">{err}</p>}
    </div>
  );
}

function Detail({ c, onClose }) {
  const v = VERDICT[c.verdict];
  const found = c.tracks.filter(isReal);
  const filtered = c.tracks.filter((t) => isSolid(t) && t.likely_reflection);
  return (
    <div className="fixed inset-0 z-10 flex justify-end bg-black/30" onClick={onClose}>
      <div className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div><h2 className="text-xl font-bold">{c.id}</h2>
            <p className="font-mono text-sm tracking-wider">{c.chassis_no || "No chassis number"}</p>
            <p className="text-sm text-slate-500 break-all">{c.vehicle} · {when(c.created_at)}</p></div>
          <button onClick={onClose} className="text-2xl leading-none text-slate-400">×</button>
        </div>

        {c.prior_inspections > 0 && (
          <p className={`mt-4 rounded-lg border p-3 text-sm ${c.prior_damaged ? "border-red-300 bg-red-50 text-red-700" : "border-slate-200 bg-slate-50 text-slate-600"}`}>
            {c.prior_damaged
              ? `Repeat submission: this chassis was already flagged DAMAGED in ${c.prior_damaged}.`
              : `This chassis was inspected ${c.prior_inspections} time(s) before.`}</p>)}
        {c.status === "processing" && <p className="mt-4 rounded-lg bg-blue-50 p-3 text-sm text-blue-700">Analysing video… this page updates automatically.</p>}
        {c.status === "failed" && <p className="mt-4 rounded-lg bg-slate-100 p-3 text-sm">Inspection failed: {c.error}</p>}
        {c.status === "done" && v && (
          <div className={`mt-4 rounded-xl border p-4 ${v.banner}`}>
            <div className="text-2xl font-extrabold">{v.label}</div>
            <div className="mt-1 text-sm font-semibold">{c.action}{c.confidence != null && ` · ${pct(c.confidence)} confidence`}</div>
            <p className="mt-2 text-sm text-slate-700">{c.reason}</p>
          </div>
        )}

        {found.length > 0 && (<>
          <h3 className="mt-6 text-xs font-bold uppercase tracking-wider text-slate-500">Damage found ({found.length})</h3>
          <table className="mt-2 w-full text-sm">
            <thead className="text-left text-xs text-slate-500"><tr><th>#</th><th>Type</th><th>Seen in</th><th>Confidence</th></tr></thead>
            <tbody>{found.map((t) => (
              <tr key={t.track_id} className="border-t"><td className="py-2">{t.track_id}</td>
                <td className="capitalize">{t.damage}</td><td>{t.image_name ? t.image_name : `${t.n_frames} frames`}</td><td>{pct(t.max_conf)}</td></tr>))}</tbody>
          </table>
          <div className="mt-4 grid gap-3">{found.filter((t) => t.evidence_image).map((t) => (
            <figure key={t.track_id}>
              <img className="rounded-lg border" src={`/evidence/${t.evidence_image}`} alt={`${t.damage} evidence`} />
              <figcaption className="text-xs text-slate-500">Best frame for damage #{t.track_id}: {t.damage}</figcaption></figure>))}</div>
        </>)}

        {filtered.length > 0 && (<>
          <h3 className="mt-6 text-xs font-bold uppercase tracking-wider text-amber-700">Filtered out as possible reflections ({filtered.length}, not counted as damage)</h3>
          <div className="mt-2 grid gap-3">{filtered.map((t) => (
            <div key={t.track_id} className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm">
              <div className="font-semibold capitalize">#{t.track_id} {t.damage} <span className="font-normal text-slate-500">· detector {pct(t.max_conf)}{t.real_prob != null && ` · real-damage score ${pct(t.real_prob)}`}</span></div>
              <ul className="mt-1 list-disc pl-5 text-slate-600">{t.flags.map((f) => <li key={f}>{f}</li>)}</ul>
              {t.evidence_image && <img className="mt-2 rounded border" src={`/evidence/${t.evidence_image}`} alt="filtered detection" />}
            </div>))}</div>
        </>)}
      </div>
    </div>
  );
}

export default function App() {
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState({});
  const [filter, setFilter] = useState("ALL");
  const [open, setOpen] = useState(null);
  const [offline, setOffline] = useState(false);

  const load = useCallback(async () => {
    try {
      const [c, s] = await Promise.all([axios.get("/api/claims"), axios.get("/api/stats")]);
      setItems(c.data); setStats(s.data); setOffline(false);
    } catch { setOffline(true); }
  }, []);
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, [load]);

  const shown = items.filter((c) => filter === "ALL" || c.verdict === filter);
  const filters = [["ALL", "All"], ["CLEAN", "Clean"], ["DAMAGED", "Damaged"], ["UNCERTAIN", "Needs review"]];

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <header>
        <h1 className="text-3xl font-extrabold">DiVision</h1>
        <p className="text-sm text-slate-500">Pre-insurance vehicle inspection · detects existing damage before a policy is issued</p>
      </header>
      {offline && <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">Cannot reach the backend on port 8000. Start it and this page will reconnect.</p>}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Inspections" value={stats.total} color="#0f172a" />
        <Stat label="Clean" value={stats.clean} color="#15803d" />
        <Stat label="Damaged" value={stats.damaged} color="#dc2626" />
        <Stat label="Needs review" value={stats.uncertain} color="#d97706" />
      </div>
      <Upload onCreated={load} />
      <div className="flex flex-wrap gap-2">{filters.map(([k, l]) => (
        <button key={k} onClick={() => setFilter(k)}
          className={`rounded-full px-3 py-1 text-xs font-semibold ${filter === k ? "bg-slate-900 text-white" : "bg-white"}`}>{l}</button>))}</div>
      <div className="overflow-x-auto rounded-xl bg-white shadow-sm">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wider text-slate-500"><tr className="border-b">
            {["ID", "Chassis no.", "Source", "Result", "Damages", "Confidence", "Decision", "Time"].map((h) => <th key={h} className="p-3">{h}</th>)}</tr></thead>
          <tbody>{shown.map((c) => (
            <tr key={c.id} onClick={() => setOpen(c.id)} className="cursor-pointer border-b last:border-0 hover:bg-slate-50">
              <td className="p-3 font-mono">{c.id}</td>
              <td className="font-mono text-xs">{c.chassis_no || "—"}{c.prior_damaged && <span title="Previously flagged damaged" className="ml-1 text-red-600">⚠</span>}</td>
              <td className="max-w-[16rem] truncate" title={c.vehicle}>{c.vehicle || "—"}</td>
              <td><Badge c={c} /></td>
              <td>{c.status === "done" ? damageCount(c) : "—"}</td>
              <td className="font-bold">{pct(c.confidence)}</td>
              <td className={`italic ${ACTION_COLOR[c.action] || ""}`}>{c.action || "—"}</td>
              <td className="text-slate-500">{when(c.created_at)}</td></tr>))}
            {!shown.length && <tr><td colSpan="8" className="p-8 text-center text-slate-400">No inspections yet. Upload a video or photos above.</td></tr>}
          </tbody>
        </table>
      </div>
      {open && items.find((c) => c.id === open) && <Detail c={items.find((c) => c.id === open)} onClose={() => setOpen(null)} />}
    </div>
  );
}
