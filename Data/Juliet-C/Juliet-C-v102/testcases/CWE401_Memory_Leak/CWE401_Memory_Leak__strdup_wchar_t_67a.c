/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__strdup_wchar_t_67a.c
Label Definition File: CWE401_Memory_Leak__strdup.label.xml
Template File: sources-sinks-67a.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using wcsdup()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

typedef struct _CWE401_Memory_Leak__strdup_wchar_t_67_struct_type
{
    wchar_t * a;
} CWE401_Memory_Leak__strdup_wchar_t_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE401_Memory_Leak__strdup_wchar_t_67b_bad_sink(CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct);

void CWE401_Memory_Leak__strdup_wchar_t_67_bad()
{
    wchar_t * data;
    CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct;
    data = NULL;
    {
        wchar_t mystring[] = L"mystring";
        /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
        data = wcsdup(mystring);
        /* Use data */
        printWLine(data);
    }
    my_struct.a = data;
    CWE401_Memory_Leak__strdup_wchar_t_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE401_Memory_Leak__strdup_wchar_t_67b_goodG2B_sink(CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct);

static void goodG2B()
{
    wchar_t * data;
    CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct;
    data = NULL;
    {
        /* FIX: Allocate memory from the stack */
        wchar_t data_goodbuf[100];
        data = data_goodbuf;
        /* Initialize then use data */
        wcscpy(data, L"a string");
        printWLine(data);
    }
    my_struct.a = data;
    CWE401_Memory_Leak__strdup_wchar_t_67b_goodG2B_sink(my_struct);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE401_Memory_Leak__strdup_wchar_t_67b_goodB2G_sink(CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct);

static void goodB2G()
{
    wchar_t * data;
    CWE401_Memory_Leak__strdup_wchar_t_67_struct_type my_struct;
    data = NULL;
    {
        wchar_t mystring[] = L"mystring";
        /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
        data = wcsdup(mystring);
        /* Use data */
        printWLine(data);
    }
    my_struct.a = data;
    CWE401_Memory_Leak__strdup_wchar_t_67b_goodB2G_sink(my_struct);
}

void CWE401_Memory_Leak__strdup_wchar_t_67_good()
{
    goodG2B();
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
    CWE401_Memory_Leak__strdup_wchar_t_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE401_Memory_Leak__strdup_wchar_t_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
