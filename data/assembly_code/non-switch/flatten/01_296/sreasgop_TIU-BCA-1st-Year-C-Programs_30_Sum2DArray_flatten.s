	.file	"sreasgop_TIU-BCA-1st-Year-C-Programs_30_Sum2DArray_flatten.c"
	.text
	.globl	_TIG_IZ_Kglg_argv
	.bss
	.align 8
	.type	_TIG_IZ_Kglg_argv, @object
	.size	_TIG_IZ_Kglg_argv, 8
_TIG_IZ_Kglg_argv:
	.zero	8
	.globl	_TIG_IZ_Kglg_argc
	.align 4
	.type	_TIG_IZ_Kglg_argc, @object
	.size	_TIG_IZ_Kglg_argc, 4
_TIG_IZ_Kglg_argc:
	.zero	4
	.globl	_TIG_IZ_Kglg_envp
	.align 8
	.type	_TIG_IZ_Kglg_envp, @object
	.size	_TIG_IZ_Kglg_envp, 8
_TIG_IZ_Kglg_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The sum is: %d"
.LC1:
	.string	"  %d "
.LC2:
	.string	"The array is: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Kglg_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Kglg_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Kglg_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 91 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Kglg--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_Kglg_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_Kglg_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_Kglg_envp(%rip)
	nop
	movq	$8, -56(%rbp)
.L23:
	cmpq	$14, -56(%rbp)
	ja	.L26
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L26-.L8
	.long	.L16-.L8
	.long	.L26-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L26-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -56(%rbp)
	jmp	.L17
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L24
	jmp	.L25
.L10:
	cmpl	$2, -64(%rbp)
	jg	.L19
	movq	$3, -56(%rbp)
	jmp	.L17
.L19:
	movq	$1, -56(%rbp)
	jmp	.L17
.L11:
	movq	$7, -56(%rbp)
	jmp	.L17
.L16:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -68(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L17
.L15:
	movl	-64(%rbp), %eax
	movslq	%eax, %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movl	-48(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-64(%rbp), %eax
	movslq	%eax, %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movl	-48(%rbp,%rax,4), %eax
	addl	%eax, -60(%rbp)
	addl	$1, -64(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L17
.L9:
	cmpl	$2, -68(%rbp)
	jg	.L21
	movq	$5, -56(%rbp)
	jmp	.L17
.L21:
	movq	$4, -56(%rbp)
	jmp	.L17
.L13:
	movl	$0, -64(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L17
.L12:
	movl	$1, -48(%rbp)
	movl	$2, -44(%rbp)
	movl	$3, -40(%rbp)
	movl	$4, -36(%rbp)
	movl	$5, -32(%rbp)
	movl	$6, -28(%rbp)
	movl	$7, -24(%rbp)
	movl	$8, -20(%rbp)
	movl	$9, -16(%rbp)
	movl	$0, -60(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -68(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L17
.L26:
	nop
.L17:
	jmp	.L23
.L25:
	call	__stack_chk_fail@PLT
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
