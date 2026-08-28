import time
import json
import os
from datetime import datetime

# Import Pair 3's modules
import models_registry
from BT_CM import SatQuerySpecialist

class SatQueryController:
    def __init__(self):
        self.specialist = SatQuerySpecialist()
        print(f"[*] SatQuery Agentic Controller Online. Specialist Version: {self.specialist.version}")

    def _determine_task(self, query):
        q = query.lower()
        if any(w in q for w in ["change", "before", "after", "compare", "difference"]):
            return "BITEMPORAL_CHANGE"
        if any(w in q for w in ["locate", "find", "highlight", "detect", "where"]):
            return "OBJECT_GROUNDING"
        if any(w in q for w in ["sar", "radar", "clouds", "night", "water"]):
            return "CROSS_MODAL_SAR"
        return "SINGLE_VQA"

    def process_query(self, query, image_paths):
        """
        The Main Workflow:
        1. Route to Task
        2. Execute Specialist Logic (Math/Signal Processing)
        3. Execute Cognitive Logic (VLM Reasoning)
        4. Package Evidence
        """
        start_time = time.time()
        task = self._determine_task(query)
        num_images = len(image_paths)

        # --- VALIDATION ---
        if task in ["BITEMPORAL_CHANGE", "CROSS_MODAL_SAR"] and num_images < 2:
            return {"status": "error", "message": f"{task} requires 2 images. Only 1 provided."}

        try:
            # --- EXECUTION PATHS ---
            if task == "BITEMPORAL_CHANGE":
                # 1. Run Specialist Math (BT_CM)
                t0, t1, mask, reg_conf = self.specialist.run_bi_temporal_analysis(image_paths[0], image_paths[1])
                report = self.specialist.generate_agentic_report(mask, reg_conf)
                
                # 2. Ground the AI with the Math (Force Evidence-Grounded Response)
                grounded_prompt = (
                    f"Analyze these two satellite images. I have calculated a {report['intensity']} "
                    f"geographic change with a system trust score of {report['metrics']['system_total_trust']}. "
                    f"The signal-to-noise ratio is {report['metrics']['spatial_snr']}. "
                    f"Based on this data, describe what specifically changed in the scene."
                )
                # Note: We send t1 (current state) to the VLM to describe the change
                vlm_result = models_registry.run_vqa(image_paths[1], grounded_prompt)
                
                # 3. Get Visual Evidence (Maps/SAR/Saliency)
                sal, sar, edges, synth = self.specialist.generate_visual_modalities(t0, t1, mask)
                
                response = {
                    "text": vlm_result["text_response"],
                    "analytical_data": report,
                    "visuals": {"mask": mask, "saliency": sal, "sar_proxy": sar, "fused": synth}
                }

            elif task == "OBJECT_GROUNDING":
                # Direct call to Person 5's Grounding tool
                target = query.lower().replace("locate", "").replace("find", "").strip()
                response = models_registry.run_grounding(image_paths[0], target)

            elif task == "CROSS_MODAL_SAR":
                # Use Specialist to extract SAR water/structure features
                img = self.specialist.load_image(image_paths[0])
                _, sar, _, _ = self.specialist.generate_visual_modalities(img, img, np.zeros(img.shape[:2]))
                vlm_result = models_registry.run_vqa(image_paths[0], f"Using this SAR-enhanced view, {query}")
                response = {"text": vlm_result["text_response"], "visuals": {"sar_proxy": sar}}

            else: # SINGLE_VQA
                response = models_registry.run_vqa(image_paths[0], query)

            # --- AUDIT TRACE (For the Judges) ---
            trace = {
                "task": task,
                "engine_metadata": models_registry.get_execution_metadata(),
                "specialist_stats": report["metrics"] if task == "BITEMPORAL_CHANGE" else "N/A",
                "latency": f"{time.time() - start_time:.2f}s",
                "timestamp": datetime.now().isoformat()
            }

            return {"status": "success", "data": response, "trace": trace}

        except Exception as e:
            return {"status": "error", "message": str(e)}