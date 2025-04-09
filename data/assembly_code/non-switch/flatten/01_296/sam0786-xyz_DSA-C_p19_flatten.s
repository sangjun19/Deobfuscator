	.file	"sam0786-xyz_DSA-C_p19_flatten.c"
	.text
	.globl	_TIG_IZ_qt3V_argv
	.bss
	.align 8
	.type	_TIG_IZ_qt3V_argv, @object
	.size	_TIG_IZ_qt3V_argv, 8
_TIG_IZ_qt3V_argv:
	.zero	8
	.globl	_TIG_IZ_qt3V_envp
	.align 8
	.type	_TIG_IZ_qt3V_envp, @object
	.size	_TIG_IZ_qt3V_envp, 8
_TIG_IZ_qt3V_envp:
	.zero	8
	.globl	_TIG_IZ_qt3V_argc
	.align 4
	.type	_TIG_IZ_qt3V_argc, @object
	.size	_TIG_IZ_qt3V_argc, 4
_TIG_IZ_qt3V_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter the elements: "
	.align 8
.LC1:
	.string	"Element %d not found in the array\n"
.LC2:
	.string	"%d"
.LC3:
	.string	"Element %d found at index %d\n"
.LC4:
	.string	"Enter the element to search: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_qt3V_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qt3V_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qt3V_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qt3V--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_qt3V_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_qt3V_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_qt3V_envp(%rip)
	nop
	movq	$17, -40(%rbp)
.L30:
	cmpq	$19, -40(%rbp)
	ja	.L33
	movq	-40(%rbp), %rax
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
	.long	.L33-.L8
	.long	.L19-.L8
	.long	.L33-.L8
	.long	.L18-.L8
	.long	.L33-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L33-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	addl	$1, -44(%rbp)
	movq	$1, -40(%rbp)
	jmp	.L20
.L12:
	cmpl	$4, -48(%rbp)
	jg	.L21
	movq	$19, -40(%rbp)
	jmp	.L20
.L21:
	movq	$5, -40(%rbp)
	jmp	.L20
.L14:
	cmpl	$0, -52(%rbp)
	jne	.L23
	movq	$13, -40(%rbp)
	jmp	.L20
.L23:
	movq	$16, -40(%rbp)
	jmp	.L20
.L19:
	cmpl	$4, -44(%rbp)
	jg	.L25
	movq	$11, -40(%rbp)
	jmp	.L20
.L25:
	movq	$12, -40(%rbp)
	jmp	.L20
.L18:
	movl	$0, -52(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -48(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L20
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L32
.L15:
	movl	-44(%rbp), %eax
	cltq
	movl	-32(%rbp,%rax,4), %edx
	movl	-56(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L28
	movq	$6, -40(%rbp)
	jmp	.L20
.L28:
	movq	$18, -40(%rbp)
	jmp	.L20
.L13:
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -40(%rbp)
	jmp	.L20
.L7:
	leaq	-32(%rbp), %rdx
	movl	-48(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -48(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L20
.L10:
	movq	$3, -40(%rbp)
	jmp	.L20
.L16:
	movl	$1, -52(%rbp)
	movl	-56(%rbp), %eax
	movl	-44(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L20
.L17:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -44(%rbp)
	movq	$1, -40(%rbp)
	jmp	.L20
.L33:
	nop
.L20:
	jmp	.L30
.L32:
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
