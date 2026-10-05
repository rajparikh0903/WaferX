export const Skeleton = ({ className = "h-24" }: { className?: string }) => <div className={`skeleton ${className}`} aria-hidden />;
export const InvestigationSkeleton = () => (
  <div className="grid gap-4 lg:grid-cols-3" role="status" aria-label="Analyzing sample">
    <Skeleton className="h-44 lg:col-span-2" /><Skeleton className="h-44" /><Skeleton className="h-64 lg:col-span-3" /><Skeleton className="h-72 lg:col-span-3" /></div>);
