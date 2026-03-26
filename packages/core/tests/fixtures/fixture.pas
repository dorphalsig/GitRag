{ Captures 'unit' (Package). }
unit MyPascalUnit;

interface

type
  TMyEnum = (
    { Captures 'declEnumValue' (EnumMember). }
    EnumAlpha,
    { Captures 'declEnumValue' (EnumMember). }
    EnumBeta
  );

  { Captures 'declType' (Type). }
  TMyPascalClass = class
  private
    { Captures 'declField' (Field). }
    FMyField: Integer;
  public
    { Captures 'defProc' (Method). }
    procedure DoSomething;
    { Captures 'defProc' (Method) with massive string. }
    procedure MassiveStringProcedure;
  end;

const
  { Captures 'declConst' (Field). }
  MY_CONST = 42;

var
  { Captures 'declVar' (Field). }
  GlobalVar: Integer;

implementation

procedure TMyPascalClass.DoSomething;
begin
  FMyField := 1;
end;

procedure TMyPascalClass.MassiveStringProcedure;
var
  MassiveString: string;
begin
  MassiveString := 'This string is designed to be exceedingly long to force the chunker to split by size. ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] ' +
  'Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]';
end;

{ Captures 'initialization' (Initializer). }
initialization
  GlobalVar := 0;

{ Captures 'finalization' (Initializer). }
finalization
  GlobalVar := -1;

end.
