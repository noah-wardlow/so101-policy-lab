import { useState, useEffect, useCallback } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

const _raycaster = new THREE.Raycaster();
const _mouse = new THREE.Vector2();

/**
 * Double-click to select a MuJoCo body. Returns the selected bodyId (or null).
 */
export function useClickSelect() {
  const { gl, camera, scene } = useThree();
  const [selectedBodyId, setSelectedBodyId] = useState<number | null>(null);

  const onDblClick = useCallback(
    (e: MouseEvent) => {
      const rect = gl.domElement.getBoundingClientRect();
      _mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      _mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      _raycaster.setFromCamera(_mouse, camera);
      const hits = _raycaster.intersectObjects(scene.children, true);

      for (const hit of hits) {
        let obj: THREE.Object3D | null = hit.object;
        while (obj) {
          const bodyId = obj.userData?.bodyID;
          if (bodyId != null && bodyId > 0) {
            setSelectedBodyId((prev) => (prev === bodyId ? null : bodyId));
            return;
          }
          obj = obj.parent;
        }
      }
      // Clicked empty space — deselect
      setSelectedBodyId(null);
    },
    [gl, camera, scene],
  );

  useEffect(() => {
    gl.domElement.addEventListener('dblclick', onDblClick);
    return () => gl.domElement.removeEventListener('dblclick', onDblClick);
  }, [gl, onDblClick]);

  return selectedBodyId;
}
