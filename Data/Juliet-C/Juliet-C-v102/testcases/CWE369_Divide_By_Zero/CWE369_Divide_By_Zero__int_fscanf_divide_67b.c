/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE369_Divide_By_Zero__int_fscanf_divide_67b.c
Label Definition File: CWE369_Divide_By_Zero__int.label.xml
Template File: sources-sink-67b.tmpl.c
*/
/*
 * @description
 * CWE: 369 Divide By Zero
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Non-zero
 * Sinks: divide
 *    BadSink : Divide a constant by data
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

typedef struct _CWE369_Divide_By_Zero__int_fscanf_divide_67_struct_type
{
    int a;
} CWE369_Divide_By_Zero__int_fscanf_divide_67_struct_type;

#ifndef OMITBAD

void CWE369_Divide_By_Zero__int_fscanf_divide_67b_bad_sink(CWE369_Divide_By_Zero__int_fscanf_divide_67_struct_type my_struct)
{
    int data = my_struct.a;
    /* POTENTIAL FLAW: Possibly divide by zero */
    printIntLine(100 / data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE369_Divide_By_Zero__int_fscanf_divide_67b_goodG2B_sink(CWE369_Divide_By_Zero__int_fscanf_divide_67_struct_type my_struct)
{
    int data = my_struct.a;
    /* POTENTIAL FLAW: Possibly divide by zero */
    printIntLine(100 / data);
}

#endif /* OMITGOOD */
