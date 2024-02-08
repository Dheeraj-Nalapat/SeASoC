/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE416_Use_After_Free__malloc_free_char_54d.c
Label Definition File: CWE416_Use_After_Free__malloc_free.label.xml
Template File: sources-sinks-54d.tmpl.c
*/
/*
 * @description
 * CWE: 416 Use After Free
 * BadSource:  Allocate data using malloc(), initialize memory block, and Deallocate data using free()
 * GoodSource: Allocate data using malloc() and initialize memory block
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Use data
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE416_Use_After_Free__malloc_free_char_54e_bad_sink(char * data);

void CWE416_Use_After_Free__malloc_free_char_54d_bad_sink(char * data)
{
    CWE416_Use_After_Free__malloc_free_char_54e_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE416_Use_After_Free__malloc_free_char_54e_goodG2B_sink(char * data);

void CWE416_Use_After_Free__malloc_free_char_54d_goodG2B_sink(char * data)
{
    CWE416_Use_After_Free__malloc_free_char_54e_goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE416_Use_After_Free__malloc_free_char_54e_goodB2G_sink(char * data);

void CWE416_Use_After_Free__malloc_free_char_54d_goodB2G_sink(char * data)
{
    CWE416_Use_After_Free__malloc_free_char_54e_goodB2G_sink(data);
}

#endif /* OMITGOOD */
