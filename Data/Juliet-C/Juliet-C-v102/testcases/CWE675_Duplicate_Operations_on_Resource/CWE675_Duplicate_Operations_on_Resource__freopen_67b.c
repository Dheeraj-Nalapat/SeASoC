/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE675_Duplicate_Operations_on_Resource__freopen_67b.c
Label Definition File: CWE675_Duplicate_Operations_on_Resource.label.xml
Template File: sources-sinks-67b.tmpl.c
*/
/*
 * @description
 * CWE: 675 Duplicate Operations on Resource
 * BadSource: freopen Open and close a file using freopen() and flose()
 * GoodSource: Open a file using fopen()
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Close the file
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

typedef struct _CWE675_Duplicate_Operations_on_Resource__freopen_67_struct_type
{
    FILE * a;
} CWE675_Duplicate_Operations_on_Resource__freopen_67_struct_type;

#ifndef OMITBAD

void CWE675_Duplicate_Operations_on_Resource__freopen_67b_bad_sink(CWE675_Duplicate_Operations_on_Resource__freopen_67_struct_type my_struct)
{
    FILE * data = my_struct.a;
    /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
    fclose(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE675_Duplicate_Operations_on_Resource__freopen_67b_goodG2B_sink(CWE675_Duplicate_Operations_on_Resource__freopen_67_struct_type my_struct)
{
    FILE * data = my_struct.a;
    /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
    fclose(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE675_Duplicate_Operations_on_Resource__freopen_67b_goodB2G_sink(CWE675_Duplicate_Operations_on_Resource__freopen_67_struct_type my_struct)
{
    FILE * data = my_struct.a;
    /* Do nothing */
    /* FIX: Don't close the file in the sink */
    ; /* empty statement needed for some flow variants */
}

#endif /* OMITGOOD */
