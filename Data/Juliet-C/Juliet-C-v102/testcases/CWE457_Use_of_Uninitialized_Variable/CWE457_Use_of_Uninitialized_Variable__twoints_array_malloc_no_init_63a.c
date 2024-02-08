/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63a.c
Label Definition File: CWE457_Use_of_Uninitialized_Variable.c_array.label.xml
Template File: sources-sinks-63a.tmpl.c
*/
/*
 * @description
 * CWE: 457 Use of Uninitialized Variable
 * BadSource: no_init Don't initialize data
 * GoodSource: Initialize data
 * Sinks: use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data
 * Flow Variant: 63 Data flow: pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_bad_sink(twoints * * data_ptr);

void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63_bad()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    /* Don't initialize data */
    ; /* empty statement needed for some flow variants */
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_bad_sink(&data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_goodG2B_sink(twoints * * data);

static void goodG2B()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    {
        int i;
        for(i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_goodG2B_sink(&data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_goodB2G_sink(twoints * * data);

static void goodB2G()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    /* Don't initialize data */
    ; /* empty statement needed for some flow variants */
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63b_goodB2G_sink(&data);
}

void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63_good()
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
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_no_init_63_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
