import { useState, useMemo } from 'react';
import type { RoomDimensions, WindowSpecs, WallType } from '../types';


interface UIWindow extends WindowSpecs {
  id: string;
}

const WALLS: WallType[] = ['South', 'West', 'North', 'East'];

export default function WindowDashboard() {
  const [activeTab, setActiveTab] = useState<WallType>('South');
  const [room, setRoom] = useState<RoomDimensions>({ width: 10, depth: 10 });
  const [windows, setWindows] = useState<UIWindow[]>([]);


  const activeWindows = useMemo(
    () => windows.filter((w) => w.wall === activeTab),
    [windows, activeTab]
  );

  const handleAddWindow = () => {
    if (activeWindows.length >= 3) return;
    
    const newWindow: UIWindow = {
      id: crypto.randomUUID(),
      wall: activeTab,
      center: room.width / 2,
      width: 2.0,
      sill: 0.8,
      head: 2.2,
    };
    
    setWindows([...windows, newWindow]);
  };

  const handleRemoveWindow = (id: string) => {
    setWindows(windows.filter((w) => w.id !== id));
  };

  const handleWindowChange = (id: string, field: keyof WindowSpecs, value: number) => {
    setWindows(
      windows.map((w) => (w.id === id ? { ...w, [field]: value } : w))
    );
  };

  return (
    <div className="p-6 max-w-4xl mx-auto font-sans">
      <h1 className="text-2xl font-bold mb-6">OmegaFlux Multi-Window Configurator</h1>


      <div className="bg-gray-100 p-4 rounded-lg mb-8 flex gap-6">
        <div>
          <label className="block text-sm font-semibold mb-1">Room Width (X-axis)</label>
          <input 
            type="range" min="3" max="20" step="0.5" value={room.width}
            onChange={(e) => setRoom({ ...room, width: parseFloat(e.target.value) })}
          />
          <span className="ml-2">{room.width}m</span>
        </div>
        <div>
          <label className="block text-sm font-semibold mb-1">Room Depth (Y-axis)</label>
          <input 
            type="range" min="3" max="20" step="0.5" value={room.depth}
            onChange={(e) => setRoom({ ...room, depth: parseFloat(e.target.value) })}
          />
          <span className="ml-2">{room.depth}m</span>
        </div>
      </div>


      <div className="flex border-b mb-6">
        {WALLS.map((wall) => (
          <button
            key={wall}
            onClick={() => setActiveTab(wall)}
            className={`px-6 py-2 font-semibold ${
              activeTab === wall 
                ? 'border-b-2 border-blue-600 text-blue-600' 
                : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            {wall} Wall
          </button>
        ))}
      </div>


      <div className="space-y-6">
        {activeWindows.length === 0 ? (
          <p className="text-gray-500 italic">No windows present on the {activeTab} wall.</p>
        ) : (
          activeWindows.map((win, index) => (
            <div key={win.id} className="border border-gray-200 p-4 rounded-lg relative bg-white">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-lg">Window {index + 1}</h3>
                <button 
                  onClick={() => handleRemoveWindow(win.id)}
                  className="text-red-500 text-sm hover:underline"
                >
                  Remove Window
                </button>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-600">Center Position</label>
                  <input 
                    type="range" min="0" max={activeTab === 'South' || activeTab === 'North' ? room.width : room.depth} step="0.1" 
                    value={win.center}
                    onChange={(e) => handleWindowChange(win.id, 'center', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-right text-xs">{win.center}m</div>
                </div>
                <div>
                  <label className="block text-sm text-gray-600">Width</label>
                  <input 
                    type="range" min="0.5" max="10" step="0.1" 
                    value={win.width}
                    onChange={(e) => handleWindowChange(win.id, 'width', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-right text-xs">{win.width}m</div>
                </div>
                <div>
                  <label className="block text-sm text-gray-600">Sill Height</label>
                  <input 
                    type="range" min="0" max="1.5" step="0.1" 
                    value={win.sill}
                    onChange={(e) => handleWindowChange(win.id, 'sill', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-right text-xs">{win.sill}m</div>
                </div>
                <div>
                  <label className="block text-sm text-gray-600">Head Height</label>
                  <input 
                    type="range" min="1.5" max="4" step="0.1" 
                    value={win.head}
                    onChange={(e) => handleWindowChange(win.id, 'head', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-right text-xs">{win.head}m</div>
                </div>
              </div>
            </div>
          ))
        )}


        {activeWindows.length < 3 && (
          <button 
            onClick={handleAddWindow}
            className="w-full py-3 border-2 border-dashed border-gray-300 text-gray-500 rounded-lg hover:border-blue-500 hover:text-blue-500 transition-colors"
          >
            + Add Window to {activeTab} Wall
          </button>
        )}
      </div>
    </div>
  );
}