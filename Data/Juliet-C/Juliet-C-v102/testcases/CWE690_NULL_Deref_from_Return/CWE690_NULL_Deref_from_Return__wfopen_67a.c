/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE690_NULL_Deref_from_Return__wfopen_67a.c
Label Definition File: CWE690_NULL_Deref_from_Return.fclose.label.xml
Template File: source-sinks-67a.tmpl.c
*/
/*
 * @description
 * CWE: 690 Unchecked Return Value To NULL Pointer
 * BadSource: wfopen Open data with wfopen()
 * Sinks: 0
 *    GoodSink: Check data for NULL
 *    BadSink : Do not check data for NULL
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define WFOPEN _wfopen
#else
/*fopen is used on unix-based OSs */
# define WFOPEN fopen
#endif

typedef struct _CWE690_NULL_Deref_from_Return__wfopen_67_struct_type
{
    FILE * a;
} CWE690_NULL_Deref_from_Return__wfopen_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE690_NULL_Deref_from_Return__wfopen_67b_bad_sink(CWE690_NULL_Deref_from_Return__wfopen_67_struct_type my_struct);

void CWE690_NULL_Deref_from_Return__wfopen_67_bad()
{
    FILE * data;
    CWE690_NULL_Deref_from_Return__wfopen_67_struct_type my_struct;
    /* Initialize data */
    data = NULL;
    data = WFOPEN(L"file.txt", L"w+");
    my_struct.a = data;
    CWE690_NULL_Deref_from_Return__wfopen_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G uses the BadSource with the GoodSink */
void CWE690_NULL_Deref_from_Return__wfopen_67b_goodB2G_sink(CWE690_NULL_Deref_from_Return__wfopen_67_struct_type my_struct);

static void goodB2G()
{
    FILE * data;
    CWE690_NULL_Deref_from_Return__wfopen_67_struct_type my_struct;
    /* Initialize data */
    data = NULL;
    data = WFOPEN(L"file.txt", L"w+");
    my_struct.a = data;
    CWE690_NULL_Deref_from_Return__wfopen_67b_goodB2G_sink(my_struct);
}

void CWE690_NULL_Deref_from_Return__wfopen_67_good()
{
    goodB2G();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE690_NULL_Deref_from_Return__wfopen_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE690_NULL_Deref_from_Return__wfopen_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif