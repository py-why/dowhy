import fs from "fs";
import path from "path";
import url from "url";
import { JSDOM } from 'jsdom'

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const currentVersion = process.env['CURRENT_VERSION'] ?? 'main'

const DOCS_ROOT = path.join(__dirname, "../../dowhy-docs")
const versionLinks = readVersionInfo()

// Patch all of the published versions
fs
	.readdirSync(DOCS_ROOT)
	.filter(d => d.startsWith("v") && d != currentVersion)
	.map(v => path.join(DOCS_ROOT, v, 'index.html'))
	.forEach(patchDocVersion);


function patchDocVersion(version) {
	console.log("updating", version)
	const content = fs.readFileSync(version, 'utf-8');
	const { window } = new JSDOM(content)
	let found = window.document.querySelector('#version_switcher_menu > dl')
	if (found == null) {
		// check old RST version (<= v0.8)
		found = window.document.querySelector('.rst-other-versions > dl')
	}

	// Remove old version links
	for (let i = found.children.length - 1; i >= 0; i--) {
		const child = found.children[i]
		if (child.tagName === 'DD') {
			found.removeChild(child)
		}
	}

	// Append updated version links
	for (const link of versionLinks) {
		found.appendChild(link)
	}
	fs.writeFileSync(version, window.document.documentElement.outerHTML)
}

/**
 * Reads the version info from the latest main on main.
 * 
 * sample result: [
 *    <dd><a href="../v0.8/index.html">v0.8</a></dd>
 *    <dd><a href="../v0.7.1/index.html">v0.7.1</a></dd>
 *	  <dd><a href="../v0.7/index.html">v0.7</a></dd>
 *  ]

 * @returns {Element[]} - the result list of version links. Each element is a 'dd' tag wrapping an 'a' tag.
 */
function readVersionInfo() {
	const content = fs.readFileSync(path.join(DOCS_ROOT, currentVersion, 'index.html'), 'utf-8');
	const versions = []
	const { window } = new JSDOM(content)
	const found = window.document.querySelector('#version_switcher_menu > dl')
	for (const child of found.children) {
		if (child.tagName === 'DD') {
			versions.push(child)
		}
	}
	return versions
}