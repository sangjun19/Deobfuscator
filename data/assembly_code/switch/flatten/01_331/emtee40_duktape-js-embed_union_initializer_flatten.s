	.file	"emtee40_duktape-js-embed_union_initializer_flatten.c"
	.text
	.globl	_TIG_IZ_awmx_argv
	.bss
	.align 8
	.type	_TIG_IZ_awmx_argv, @object
	.size	_TIG_IZ_awmx_argv, 8
_TIG_IZ_awmx_argv:
	.zero	8
	.globl	_TIG_IZ_awmx_envp
	.align 8
	.type	_TIG_IZ_awmx_envp, @object
	.size	_TIG_IZ_awmx_envp, 8
_TIG_IZ_awmx_envp:
	.zero	8
	.globl	_TIG_IZ_awmx_argc
	.align 4
	.type	_TIG_IZ_awmx_argc, @object
	.size	_TIG_IZ_awmx_argc, 4
_TIG_IZ_awmx_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Unknown type: %d\n"
.LC1:
	.string	"Double: %lf\n"
.LC2:
	.string	"String: %s\n"
.LC3:
	.string	"Integer: %d\n"
	.text
	.type	dump_struct, @function
dump_struct:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L17:
	cmpq	$8, -8(%rbp)
	ja	.L18
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L18-.L4
	.long	.L19-.L4
	.long	.L5-.L4
	.long	.L18-.L4
	.long	.L18-.L4
	.long	.L3-.L4
	.text
.L3:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L11
.L8:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L11
.L5:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L11
.L9:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$2, %eax
	je	.L12
	cmpl	$2, %eax
	jg	.L13
	testl	%eax, %eax
	je	.L14
	cmpl	$1, %eax
	je	.L15
	jmp	.L13
.L12:
	movq	$5, -8(%rbp)
	jmp	.L16
.L15:
	movq	$1, -8(%rbp)
	jmp	.L16
.L14:
	movq	$2, -8(%rbp)
	jmp	.L16
.L13:
	movq	$8, -8(%rbp)
	nop
.L16:
	jmp	.L11
.L7:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L11
.L18:
	nop
.L11:
	jmp	.L17
.L19:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	dump_struct, .-dump_struct
	.section	.rodata
.LC5:
	.string	"foo"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_awmx_envp(%rip)
	nop
.L21:
	movq	$0, _TIG_IZ_awmx_argv(%rip)
	nop
.L22:
	movl	$0, _TIG_IZ_awmx_argc(%rip)
	nop
	nop
.L23:
.L24:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-awmx--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_awmx_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_awmx_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_awmx_envp(%rip)
	nop
	movq	$2, -88(%rbp)
.L30:
	cmpq	$2, -88(%rbp)
	je	.L25
	cmpq	$2, -88(%rbp)
	ja	.L33
	cmpq	$0, -88(%rbp)
	je	.L27
	cmpq	$1, -88(%rbp)
	jne	.L33
	movl	$0, -80(%rbp)
	movl	$123, -72(%rbp)
	movl	$0, -64(%rbp)
	movl	$234, -56(%rbp)
	movl	$1, -48(%rbp)
	movsd	.LC4(%rip), %xmm0
	movsd	%xmm0, -40(%rbp)
	movl	$2, -32(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, -24(%rbp)
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	dump_struct
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	dump_struct
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	dump_struct
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	dump_struct
	movq	$0, -88(%rbp)
	jmp	.L28
.L27:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L32
.L25:
	movq	$1, -88(%rbp)
	jmp	.L28
.L33:
	nop
.L28:
	jmp	.L30
.L32:
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.long	446676599
	.long	1079958831
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
