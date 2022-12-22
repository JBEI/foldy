import { AnonymousSubject } from "rxjs-compat";

type AnyObject = { [key: string]: any }; //TODO: replace anything using this

declare module "parse-pdb" {
  export class Atom {
    serial: number;
    name: string;
    altLoc: string;
    resName: string;
    chainID: string;
    resSeq: number;
    iCode: string;
    x: number;
    y: number;
    z: number;
    occupancy: number;
    tempFactor: number;
    element: string;
    charge: string;
  }
  export class ParsedPdb {
    atoms: Atom[];
  }

  export function ParsePdb(pdb: string): ParsedPdb;

  export default ParsePdb;
}
