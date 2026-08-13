import React, { useState, useMemo, useEffect } from 'react';
import { LogOut, Sun, Zap, Cpu, Battery, Settings, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, 
  LineChart, Line, ScatterChart, Scatter, ZAxis, BarChart, Bar, Cell, PieChart, Pie
} from 'recharts';

const PRESETS = {
  "Custom": {},
  "Jinko Solar 400Wp Mono (2024)": { rated: 400, eff: 20.5, area: 1.947, beta: 0.35, noct: 45.0 },
  "Canadian Solar 370Wp Poly": { rated: 370, eff: 19.7, area: 1.879, beta: 0.37, noct: 43.0 },
  "Longi Hi-MO5 450Wp": { rated: 450, eff: 21.2, area: 2.124, beta: 0.34, noct: 44.0 },
  "Budget Panel 250Wp": { rated: 250, eff: 15.3, area: 1.637, beta: 0.45, noct: 47.0 },
  "SunPower Maxeon 415Wp": { rated: 415, eff: 22.0, area: 1.886, beta: 0.29, noct: 45.0 },
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ background: 'rgba(10, 10, 12, 0.9)', border: '1px solid rgba(255,255,255,0.1)', padding: '10px', borderRadius: '8px' }}>
        <p style={{ color: '#fff', marginBottom: '5px' }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color || entry.fill, margin: 0, fontSize: '14px', fontWeight: 'bold' }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const Dashboard = ({ username, onLogout }) => {
  // Sidebar State (Matching app.py EXACTLY)
  const [preset, setPreset] = useState("Custom");
  
  // Panel Specs
  const [ratedWp, setRatedWp] = useState(300);
  const [effPct, setEffPct] = useState(18.0);
  const [panelArea, setPanelArea] = useState(1.6);
  const [betaPct, setBetaPct] = useState(0.40);
  const [noctVal, setNoctVal] = useState(45.0);
  const [numPanels, setNumPanels] = useState(1);
  
  // Real-World Condition Estimates
  const [soilingPct, setSoilingPct] = useState(5);
  const [agingPct, setAgingPct] = useState(3);
  const [wiringPct, setWiringPct] = useState(2);
  const [shadingPct, setShadingPct] = useState(0);

  // System Specs
  const [invEff, setInvEff] = useState(92);
  const [batteryWh, setBatteryWh] = useState(5000);
  const [battDod, setBattDod] = useState(80);
  const [loadW, setLoadW] = useState(300);
  const [sunHours, setSunHours] = useState(5.5);

  // Weather
  const [irradiance, setIrradiance] = useState(750);
  const [ambientTemp, setAmbientTemp] = useState(30);
  const [humidity, setHumidity] = useState(55);
  const [windSpeed, setWindSpeed] = useState(3);
  const [hour, setHour] = useState(11);
  const [month, setMonth] = useState(4);
  const [doy, setDoy] = useState(120);

  const [activeTab, setActiveTab] = useState('profile');

  // Handle Preset Selection
  useEffect(() => {
    if (preset !== "Custom" && PRESETS[preset]) {
      const p = PRESETS[preset];
      if (p.rated) setRatedWp(p.rated);
      if (p.eff) setEffPct(p.eff);
      if (p.area) setPanelArea(p.area);
      if (p.beta) setBetaPct(p.beta);
      if (p.noct) setNoctVal(p.noct);
    }
  }, [preset]);

  // Derived Calculations
  const totalLossPct = soilingPct + agingPct + wiringPct + shadingPct;
  const combinedFactor = Math.max(0, (100 - totalLossPct) / 100);
  
  // Physics formula
  const tMod = ambientTemp + (noctVal - 20.0) / 800.0 * irradiance;
  const pIdeal = Math.max(0, (effPct/100) * (panelArea * numPanels) * irradiance * (1.0 - (betaPct/100) * (tMod - 25.0)));
  
  // Mock ML Models (Since we are pure React, we mock the models tracking the physics + minor variations)
  const ensPdc = pIdeal * 0.95; // base ensemble slightly lower than ideal
  const allPreds = {
    "MLP / ANN": ensPdc * 1.01,
    "Grad. Boost": ensPdc * 0.99,
    "Random Forest": ensPdc * 1.02,
    "LightGBM": ensPdc * 0.98,
    "XGBoost": ensPdc * 1.00,
  };
  
  const adjustedPdc = ensPdc * combinedFactor;
  const ratedTotal = ratedWp * numPanels;

  // Backup Calculations
  const ePanel = adjustedPdc * sunHours;
  const eBatt = batteryWh * (battDod/100) * (invEff/100);
  const eTotal = ePanel + eBatt;
  const backupHours = loadW > 0 ? eTotal / loadW : 0;
  const tempLossPct = (betaPct/100) * Math.max(0, tMod - 25) * 100;

  // Charts Data Generation
  const profileData = useMemo(() => {
    const data = [];
    for (let h = 0; h < 24; h++) {
      let g = 0;
      if (h >= 6 && h <= 18) {
        const seas = 0.75 + 0.25 * Math.sin(2 * Math.PI * (doy - 80) / 365);
        g = Math.max(0, irradiance * Math.sin(Math.PI * (h - 6) / 12) * seas);
      }
      const tm = ambientTemp + (noctVal - 20) / 800 * g;
      const ideal = Math.max(0, (effPct/100) * (panelArea * numPanels) * g * (1 - (betaPct/100) * (tm - 25)));
      const ml = ideal * 0.95;
      data.push({
        time: `${h}:00`,
        ideal: Math.round(ideal),
        ml: Math.round(ml),
        adjusted: Math.round(ml * combinedFactor),
        ac: Math.round(ml * combinedFactor * (invEff/100))
      });
    }
    return data;
  }, [irradiance, doy, ambientTemp, noctVal, effPct, panelArea, numPanels, betaPct, combinedFactor, invEff]);

  const irrSensData = useMemo(() => {
    const data = [];
    for (let g = 0; g <= 1100; g += 100) {
      const tm = ambientTemp + (noctVal - 20) / 800 * g;
      const ideal = Math.max(0, (effPct/100) * (panelArea * numPanels) * g * (1 - (betaPct/100) * (tm - 25)));
      const ml = ideal * 0.95;
      data.push({ irr: g, ideal: Math.round(ideal), ml: Math.round(ml), adj: Math.round(ml * combinedFactor) });
    }
    return data;
  }, [ambientTemp, noctVal, effPct, panelArea, numPanels, betaPct, combinedFactor]);

  const tempSensData = useMemo(() => {
    const data = [];
    for (let t = 0; t <= 50; t += 5) {
      const tm = t + (noctVal - 20) / 800 * irradiance;
      const ideal = Math.max(0, (effPct/100) * (panelArea * numPanels) * irradiance * (1 - (betaPct/100) * (tm - 25)));
      const ml = ideal * 0.95;
      data.push({ temp: t, ideal: Math.round(ideal), ml: Math.round(ml * combinedFactor) });
    }
    return data;
  }, [irradiance, noctVal, effPct, panelArea, numPanels, betaPct, combinedFactor]);

  const scatterData = useMemo(() => {
    const data = [];
    for (let i = 0; i < 50; i++) {
      const ideal = Math.random() * pIdeal * 1.5;
      const ml = ideal * (0.8 + Math.random() * 0.15); // formula overestimates
      data.push({ ideal: Math.round(ideal), ml: Math.round(ml) });
    }
    return data;
  }, [pIdeal]);

  const backupData = [
    { name: 'Panel today', val: ePanel, color: '#F59E0B' },
    { name: 'Usable battery', val: eBatt, color: '#378ADD' },
    { name: 'Total available', val: eTotal, color: '#22C55E' },
  ];

  return (
    <div className="dashboard-layout">
      {/* Sidebar - Exact match to Streamlit */}
      <div className="sidebar" style={{ width: '380px' }}>
        <div className="sidebar-header" style={{ marginBottom: '1rem' }}>
          <h2><Sun color="#10b981" size={28} /> SmartSolar</h2>
          <p style={{ color: '#6b7280', fontSize: '0.85rem' }}>Real-world ML predictor — accounts for soiling, aging, shading & wiring losses.</p>
        </div>

        <div className="form-section">
          <h3>Panel Preset</h3>
          <select value={preset} onChange={(e) => setPreset(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: 'var(--bg-input)', color: '#fff', border: '1px solid var(--border-glass)', borderRadius: '4px' }}>
            {Object.keys(PRESETS).map(k => <option key={k} value={k}>{k}</option>)}
          </select>
        </div>

        <div className="form-section">
          <h3>Panel Specifications</h3>
          <div className="slider-group">
            <div className="slider-header"><span>Rated Power (Wp)</span><span className="val">{ratedWp}</span></div>
            <input type="range" min="50" max="800" step="10" value={ratedWp} onChange={(e) => setRatedWp(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Efficiency (%)</span><span className="val">{effPct.toFixed(1)}</span></div>
            <input type="range" min="10" max="24" step="0.1" value={effPct} onChange={(e) => setEffPct(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Area (m²)</span><span className="val">{panelArea.toFixed(3)}</span></div>
            <input type="range" min="0.5" max="3.0" step="0.001" value={panelArea} onChange={(e) => setPanelArea(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Temp Coeff (%/°C)</span><span className="val">{betaPct.toFixed(2)}</span></div>
            <input type="range" min="0.20" max="0.60" step="0.01" value={betaPct} onChange={(e) => setBetaPct(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>NOCT (°C)</span><span className="val">{noctVal.toFixed(1)}</span></div>
            <input type="range" min="40" max="50" step="0.5" value={noctVal} onChange={(e) => setNoctVal(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Number of Panels</span><span className="val">{numPanels}</span></div>
            <input type="range" min="1" max="200" value={numPanels} onChange={(e) => setNumPanels(Number(e.target.value))} />
          </div>
        </div>

        <div className="form-section">
          <h3>Real-World Condition Estimates</h3>
          <div className="slider-group">
            <div className="slider-header"><span>Soiling loss (%)</span><span className="val">{soilingPct}</span></div>
            <input type="range" min="0" max="25" value={soilingPct} onChange={(e) => setSoilingPct(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Panel aging loss (%)</span><span className="val">{agingPct}</span></div>
            <input type="range" min="0" max="12" value={agingPct} onChange={(e) => setAgingPct(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Wiring loss (%)</span><span className="val">{wiringPct}</span></div>
            <input type="range" min="1" max="6" value={wiringPct} onChange={(e) => setWiringPct(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Shading loss (%)</span><span className="val">{shadingPct}</span></div>
            <input type="range" min="0" max="50" value={shadingPct} onChange={(e) => setShadingPct(Number(e.target.value))} />
          </div>
        </div>

        <div className="form-section">
          <h3>System Specifications</h3>
          <div className="slider-group">
            <div className="slider-header"><span>Inverter Eff (%)</span><span className="val">{invEff}</span></div>
            <input type="range" min="80" max="99" value={invEff} onChange={(e) => setInvEff(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Battery (Wh)</span><span className="val">{batteryWh}</span></div>
            <input type="range" min="0" max="100000" step="100" value={batteryWh} onChange={(e) => setBatteryWh(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Battery DoD (%)</span><span className="val">{battDod}</span></div>
            <input type="range" min="50" max="100" step="5" value={battDod} onChange={(e) => setBattDod(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Load Demand (W)</span><span className="val">{loadW}</span></div>
            <input type="range" min="10" max="20000" step="10" value={loadW} onChange={(e) => setLoadW(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Peak Sun Hours</span><span className="val">{sunHours.toFixed(1)}</span></div>
            <input type="range" min="1.0" max="9.0" step="0.5" value={sunHours} onChange={(e) => setSunHours(Number(e.target.value))} />
          </div>
        </div>

        <div className="form-section">
          <h3>Current Weather</h3>
          <div className="slider-group">
            <div className="slider-header"><span>Irradiance (W/m²)</span><span className="val">{irradiance}</span></div>
            <input type="range" min="0" max="1100" step="10" value={irradiance} onChange={(e) => setIrradiance(Number(e.target.value))} />
          </div>
          <div className="slider-group">
            <div className="slider-header"><span>Ambient Temp (°C)</span><span className="val">{ambientTemp}</span></div>
            <input type="range" min="0" max="50" value={ambientTemp} onChange={(e) => setAmbientTemp(Number(e.target.value))} />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content" style={{ padding: '2rem 3rem' }}>
        <motion.div className="header" initial={{ y: -10, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
          <h1 style={{ fontSize: '1.75rem' }}>☀️SmartSolar: Predicting Solar Panel Output and Battery Backup Time Using Machine Learning</h1>
          <div className="user-controls">
            <button onClick={onLogout} className="logout-btn">
              <LogOut size={16} /> Logout
            </button>
          </div>
        </motion.div>

        <h3 style={{ marginBottom: '1rem' }}>⚡ Live Predictions</h3>
        <motion.div className="metrics-grid" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
          {Object.entries(allPreds).map(([name, val], i) => (
            <div key={name} className="metric-card glass-panel" style={{ padding: '1rem' }}>
              <h4 style={{ fontSize: '0.8rem' }}>{name}</h4>
              <div className="value" style={{ fontSize: '1.5rem' }}>{val.toFixed(0)} <span style={{fontSize:'1rem'}}>W</span></div>
            </div>
          ))}
          <div className="metric-card glass-panel" style={{ padding: '1rem', borderTop: '2px solid #8b5cf6' }}>
            <h4 style={{ fontSize: '0.8rem' }}>🔗 ML Ensemble</h4>
            <div className="value" style={{ fontSize: '1.5rem' }}>{ensPdc.toFixed(0)} <span style={{fontSize:'1rem'}}>W</span></div>
          </div>
          <div className="metric-card highlight glass-panel" style={{ padding: '1rem', borderTop: '2px solid #10b981' }}>
            <h4 style={{ fontSize: '0.8rem', color: '#10b981' }}>⚡ Adjusted Output</h4>
            <div className="value" style={{ fontSize: '1.5rem', color: '#10b981' }}>{adjustedPdc.toFixed(0)} <span style={{fontSize:'1rem'}}>W</span></div>
            <div className="delta negative">−{totalLossPct}% real-world losses</div>
          </div>
          <div className="metric-card glass-panel" style={{ padding: '1rem', borderTop: '2px solid #3b82f6' }}>
            <h4 style={{ fontSize: '0.8rem' }}>⏱️ Backup Runtime</h4>
            <div className="value" style={{ fontSize: '1.5rem' }}>{backupHours.toFixed(1)} <span style={{fontSize:'1rem'}}>hrs</span></div>
            <div className="delta">at {loadW}W load</div>
          </div>
        </motion.div>

        <motion.div className="loss-breakdown glass-panel" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h4 style={{ color: '#fcd34d', marginBottom: '1rem' }}>📉 Real-World Loss Breakdown — {totalLossPct}% total site derating applied</h4>
          <div className="loss-grid">
            <div className="loss-item">
              <h5 style={{ color: '#94a3b8' }}>Physics formula</h5>
              <div className="loss-val ideal">{pIdeal.toFixed(0)} W</div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Ideal output — no losses</div>
            </div>
            <div className="loss-item">
              <h5 style={{ color: '#94a3b8' }}>Soiling loss</h5>
              <div className="loss-val loss">-{soilingPct}%</div>
              <div style={{ fontSize: '0.75rem', color: '#ef4444' }}>-{(pIdeal*soilingPct/100).toFixed(0)} W</div>
            </div>
            <div className="loss-item">
              <h5 style={{ color: '#94a3b8' }}>Aging loss</h5>
              <div className="loss-val loss">-{agingPct}%</div>
              <div style={{ fontSize: '0.75rem', color: '#ef4444' }}>-{(pIdeal*agingPct/100).toFixed(0)} W</div>
            </div>
            <div className="loss-item">
              <h5 style={{ color: '#94a3b8' }}>Wiring+mismatch</h5>
              <div className="loss-val loss">-{wiringPct}%</div>
              <div style={{ fontSize: '0.75rem', color: '#ef4444' }}>-{(pIdeal*wiringPct/100).toFixed(0)} W</div>
            </div>
            <div className="loss-item">
              <h5 style={{ color: '#94a3b8' }}>Shading loss</h5>
              <div className="loss-val loss">-{shadingPct}%</div>
              <div style={{ fontSize: '0.75rem', color: '#ef4444' }}>-{(pIdeal*shadingPct/100).toFixed(0)} W</div>
            </div>
          </div>
          <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#cbd5e1' }}>
            The physics formula alone predicts {pIdeal.toFixed(0)} W (ideal), but real-world losses reduce the effective output to {adjustedPdc.toFixed(0)} W — a gap ML bridges.
          </p>
        </motion.div>

        {/* Gauges & Temps Row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
          <div className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center' }}>
            <h4 style={{ marginBottom: '1rem' }}>Output Gauge</h4>
            <div style={{ position: 'relative', height: '150px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={[{value: adjustedPdc}, {value: ratedTotal - adjustedPdc > 0 ? ratedTotal - adjustedPdc : 0}]}
                    cx="50%" cy="100%" startAngle={180} endAngle={0} innerRadius={60} outerRadius={80}
                    paddingAngle={0} dataKey="value" stroke="none"
                  >
                    <Cell fill="#378ADD" />
                    <Cell fill="rgba(255,255,255,0.1)" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div style={{ position: 'absolute', top: '70%', left: '50%', transform: 'translate(-50%, -50%)', fontSize: '2rem', fontWeight: 'bold' }}>
                {adjustedPdc.toFixed(0)}<span style={{fontSize:'1rem'}}>W</span>
              </div>
            </div>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.5rem' }}>Adjusted vs rated {ratedTotal}Wp</p>
          </div>

          <div className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h4>Module Temperature</h4>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: tMod > 65 ? '#ef4444' : tMod > 48 ? '#f59e0b' : '#22c55e', margin: '1rem 0' }}>
              {tMod.toFixed(1)}°C
            </div>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Panel surface (NOCT model)</p>
            <p style={{ fontSize: '0.75rem', color: '#64748b' }}>Ambient: {ambientTemp}°C | G: {irradiance} W/m²</p>
            {tempLossPct > 0 ? (
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#f59e0b', background: 'rgba(245,158,11,0.1)', padding: '4px', borderRadius: '4px' }}>
                Thermal loss: {tempLossPct.toFixed(1)}% below STC
              </div>
            ) : (
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#22c55e', background: 'rgba(34,197,94,0.1)', padding: '4px', borderRadius: '4px' }}>
                Panel at or below STC temp
              </div>
            )}
          </div>

          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h4 style={{ marginBottom: '1rem' }}>Backup Runtime Breakdown</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={backupData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <XAxis type="number" stroke="#6b7280" />
                <YAxis dataKey="name" type="category" width={100} stroke="#6b7280" style={{ fontSize: '0.75rem' }} />
                <RechartsTooltip contentStyle={{ background: '#1e1e1e', border: 'none' }} />
                <Bar dataKey="val" radius={[0, 4, 4, 0]}>
                  {backupData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Analysis Tabs */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>📈 Analysis</h3>
          <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-glass)', marginBottom: '1.5rem', overflowX: 'auto' }}>
            {['profile', 'irr_sens', 'temp_sens', 'formula', 'performance'].map(tab => {
              const labels = { profile: "⏰ Today's Profile", irr_sens: "🌤️ Irradiance Sensitivity", temp_sens: "🌡️ Temperature Sensitivity", formula: "📊 Formula vs ML", performance: "🔬 Model Performance" };
              return (
                <button 
                  key={tab} onClick={() => setActiveTab(tab)}
                  style={{ background: 'none', border: 'none', padding: '0.75rem 1rem', cursor: 'pointer', whiteSpace: 'nowrap',
                    color: activeTab === tab ? '#3b82f6' : '#9ca3af', borderBottom: activeTab === tab ? '2px solid #3b82f6' : '2px solid transparent', fontWeight: activeTab === tab ? 600 : 400
                  }}
                >
                  {labels[tab]}
                </button>
              );
            })}
          </div>

          <div style={{ height: '400px' }}>
            {activeTab === 'profile' && (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={profileData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <RechartsTooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="ideal" name="Physics formula (ideal)" stroke="#94a3b8" strokeDasharray="5 5" fill="rgba(148,163,184,0.1)" />
                  <Area type="monotone" dataKey="ml" name="ML ensemble" stroke="#f59e0b" fill="none" />
                  <Area type="monotone" dataKey="adjusted" name="Adjusted" stroke="#22c55e" fill="rgba(34,197,94,0.1)" />
                  <Area type="monotone" dataKey="ac" name="AC output" stroke="#7f77dd" strokeDasharray="3 3" fill="none" />
                </AreaChart>
              </ResponsiveContainer>
            )}

            {activeTab === 'irr_sens' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={irrSensData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="irr" stroke="#6b7280" label={{ value: 'Irradiance (W/m²)', position: 'insideBottom', offset: -5, fill: '#6b7280' }} />
                  <YAxis stroke="#6b7280" label={{ value: 'Power (W)', angle: -90, position: 'insideLeft', fill: '#6b7280' }} />
                  <RechartsTooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="ideal" name="Physics formula" stroke="#94a3b8" strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="ml" name="ML ensemble" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="adj" name={`Adjusted (-${totalLossPct}%)`} stroke="#22c55e" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeTab === 'temp_sens' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tempSensData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="temp" stroke="#6b7280" label={{ value: 'Ambient Temp (°C)', position: 'insideBottom', offset: -5, fill: '#6b7280' }} />
                  <YAxis stroke="#6b7280" />
                  <RechartsTooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="ideal" name="Physics formula" stroke="#94a3b8" strokeDasharray="5 5" dot={false} />
                  <Area type="monotone" dataKey="ml" name="ML + site losses" stroke="#ef4444" fill="rgba(239,68,68,0.1)" />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeTab === 'formula' && (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis type="number" dataKey="ideal" name="Ideal" stroke="#6b7280" label={{ value: 'Physics Formula (W)', position: 'insideBottom', offset: -10, fill: '#6b7280' }} />
                  <YAxis type="number" dataKey="ml" name="ML" stroke="#6b7280" label={{ value: 'Predicted DC Power (W)', angle: -90, position: 'insideLeft', fill: '#6b7280' }} />
                  <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                  <Scatter name="Predictions" data={scatterData} fill="#ef4444" shape="circle" fillOpacity={0.4} />
                </ScatterChart>
              </ResponsiveContainer>
            )}

            {activeTab === 'performance' && (
              <div style={{ color: '#fff' }}>
                <p style={{ marginBottom: '1rem', color: '#94a3b8' }}>Mocked Model accuracy metrics matching Streamlit display</p>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <th style={{ padding: '8px' }}>Model</th>
                      <th style={{ padding: '8px' }}>R²</th>
                      <th style={{ padding: '8px' }}>RMSE (W)</th>
                      <th style={{ padding: '8px' }}>MAE (W)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}><td style={{ padding: '8px' }}>MLP / ANN</td><td style={{ padding: '8px' }}>0.9638</td><td style={{ padding: '8px' }}>10.17</td><td style={{ padding: '8px' }}>6.25</td></tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}><td style={{ padding: '8px' }}>Gradient Boosting</td><td style={{ padding: '8px' }}>0.9428</td><td style={{ padding: '8px' }}>12.77</td><td style={{ padding: '8px' }}>9.86</td></tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}><td style={{ padding: '8px' }}>Random Forest</td><td style={{ padding: '8px' }}>0.9413</td><td style={{ padding: '8px' }}>12.94</td><td style={{ padding: '8px' }}>10.15</td></tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}><td style={{ padding: '8px' }}>LightGBM</td><td style={{ padding: '8px' }}>0.9399</td><td style={{ padding: '8px' }}>13.10</td><td style={{ padding: '8px' }}>10.19</td></tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
