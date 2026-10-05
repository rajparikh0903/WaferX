import { useEffect, useState } from "react";
import { health } from "@/lib/api/client";
export function useBackendStatus() {
  const [s, setS] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    let alive = true;
    const check = () => health().then(() => alive && setS("online")).catch(() => alive && setS("offline"));
    check(); const t = setInterval(check, 30000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  return s;
}
