/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE762_Mismatched_Memory_Management_Routines__strdup_delete_array_wchar_t_68a.cpp
Label Definition File: CWE762_Mismatched_Memory_Management_Routines__strdup_delete_array.label.xml
Template File: sources-sinks-68a.tmpl.cpp
*/
/*
 * @description
 * CWE: 762 Mismatched Memory Management Routines
 * BadSource:  Allocate data using wcsdup()
 * GoodSource: Allocate data using new []
 * Sinks:
 *    GoodSink: Deallocate data using free()
 *    BadSink : Deallocate data using delete []
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE762_Mismatched_Memory_Management_Routines__strdup_delete_array_wchar_t_68
{

wchar_t * bad_data;
wchar_t * goodG2B_data;
wchar_t * goodB2G_data;

#ifndef OMITBAD

/* bad function declaration */
void bad_sink();

void bad()
{
    wchar_t * data;
    /* Initialize data*/
    data = NULL;
    {
        wchar_t mystring[] = L"mystring";
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = wcsdup(mystring);
    }
    bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void goodG2B_sink();
void goodB2G_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    /* Initialize data*/
    data = NULL;
    /* FIX: Allocate memory from the heap using new [] */
    data = new wchar_t[100];
    goodG2B_data = data;
    goodG2B_sink();
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G()
{
    wchar_t * data;
    /* Initialize data*/
    data = NULL;
    {
        wchar_t mystring[] = L"mystring";
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = wcsdup(mystring);
    }
    goodB2G_data = data;
    goodB2G_sink();
}

void good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE762_Mismatched_Memory_Management_Routines__strdup_delete_array_wchar_t_68; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
