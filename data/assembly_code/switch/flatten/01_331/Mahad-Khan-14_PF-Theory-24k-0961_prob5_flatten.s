	.file	"Mahad-Khan-14_PF-Theory-24k-0961_prob5_flatten.c"
	.text
	.globl	_TIG_IZ_Evq5_argc
	.bss
	.align 4
	.type	_TIG_IZ_Evq5_argc, @object
	.size	_TIG_IZ_Evq5_argc, 4
_TIG_IZ_Evq5_argc:
	.zero	4
	.globl	_TIG_IZ_Evq5_argv
	.align 8
	.type	_TIG_IZ_Evq5_argv, @object
	.size	_TIG_IZ_Evq5_argv, 8
_TIG_IZ_Evq5_argv:
	.zero	8
	.globl	_TIG_IZ_Evq5_envp
	.align 8
	.type	_TIG_IZ_Evq5_envp, @object
	.size	_TIG_IZ_Evq5_envp, 8
_TIG_IZ_Evq5_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the number of supplies to add: "
.LC1:
	.string	"%d"
.LC2:
	.string	"Invalid species index!"
	.align 8
.LC3:
	.string	"Enter species index (0 to %d): "
.LC4:
	.string	"Enter supply %d: "
.LC5:
	.string	"%s"
	.text
	.globl	addSupplies
	.type	addSupplies, @function
addSupplies:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$184, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -168(%rbp)
	movl	%esi, -172(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$2, -144(%rbp)
.L25:
	cmpq	$15, -144(%rbp)
	ja	.L28
	movq	-144(%rbp), %rax
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
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L29-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L11-.L4
	.long	.L28-.L4
	.long	.L10-.L4
	.long	.L29-.L4
	.long	.L8-.L4
	.long	.L28-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-152(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-152(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rcx
	movq	-168(%rbp), %rax
	addq	%rcx, %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -136(%rbp)
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-168(%rbp), %rax
	addq	%rax, %rdx
	movq	-136(%rbp), %rax
	movq	%rax, (%rdx)
	movl	$0, -148(%rbp)
	movq	$13, -144(%rbp)
	jmp	.L18
.L3:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -144(%rbp)
	jmp	.L18
.L7:
	movl	-156(%rbp), %eax
	testl	%eax, %eax
	jns	.L19
	movq	$0, -144(%rbp)
	jmp	.L18
.L19:
	movq	$6, -144(%rbp)
	jmp	.L18
.L10:
	movl	-172(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-156(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -144(%rbp)
	jmp	.L18
.L15:
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-152(%rbp), %edx
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	-184(%rbp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	movq	$4, -144(%rbp)
	jmp	.L18
.L6:
	movl	-152(%rbp), %eax
	cmpl	%eax, -148(%rbp)
	jge	.L21
	movq	$10, -144(%rbp)
	jmp	.L18
.L21:
	movq	$1, -144(%rbp)
	jmp	.L18
.L11:
	movl	-156(%rbp), %eax
	cmpl	%eax, -172(%rbp)
	jg	.L23
	movq	$15, -144(%rbp)
	jmp	.L18
.L23:
	movq	$14, -144(%rbp)
	jmp	.L18
.L8:
	movl	-148(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-184(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %ecx
	movl	-148(%rbp), %eax
	addl	%ecx, %eax
	cltq
	salq	$3, %rax
	leaq	(%rdx,%rax), %rbx
	leaq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, (%rbx)
	addl	$1, -148(%rbp)
	movq	$13, -144(%rbp)
	jmp	.L18
.L16:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -144(%rbp)
	jmp	.L18
.L14:
	movq	$8, -144(%rbp)
	jmp	.L18
.L28:
	nop
.L18:
	jmp	.L25
.L29:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L27
	call	__stack_chk_fail@PLT
.L27:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	addSupplies, .-addSupplies
	.globl	removeSpecies
	.type	removeSpecies, @function
removeSpecies:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	%ecx, -60(%rbp)
	movq	$2, -24(%rbp)
.L47:
	cmpq	$13, -24(%rbp)
	ja	.L48
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L33(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L33(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L33:
	.long	.L48-.L33
	.long	.L40-.L33
	.long	.L39-.L33
	.long	.L48-.L33
	.long	.L48-.L33
	.long	.L38-.L33
	.long	.L37-.L33
	.long	.L36-.L33
	.long	.L49-.L33
	.long	.L34-.L33
	.long	.L48-.L33
	.long	.L48-.L33
	.long	.L48-.L33
	.long	.L32-.L33
	.text
.L40:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-48(%rbp), %rax
	movl	%edx, (%rax)
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-56(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$8, -24(%rbp)
	jmp	.L42
.L34:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	addq	$1, %rax
	salq	$3, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	(%rcx), %rax
	movq	%rax, (%rdx)
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	addq	$1, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	addl	$1, -28(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L42
.L32:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-60(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	-60(%rbp), %eax
	movl	%eax, -28(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L42
.L37:
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movl	-60(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jge	.L43
	movq	$7, -24(%rbp)
	jmp	.L42
.L43:
	movq	$13, -24(%rbp)
	jmp	.L42
.L38:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	cmpl	%eax, -28(%rbp)
	jge	.L45
	movq	$9, -24(%rbp)
	jmp	.L42
.L45:
	movq	$1, -24(%rbp)
	jmp	.L42
.L36:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-60(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	addl	$1, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L42
.L39:
	movl	$0, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L42
.L48:
	nop
.L42:
	jmp	.L47
.L49:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	removeSpecies, .-removeSpecies
	.section	.rodata
.LC6:
	.string	"Exiting..."
.LC7:
	.string	"Invalid choice! Try again."
	.align 8
.LC8:
	.string	"Enter supply index (0 to %d): "
	.align 8
.LC9:
	.string	"\n--- Pets in Heart Inventory Menu ---"
.LC10:
	.string	"1. Add Supplies"
.LC11:
	.string	"2. Update Supply"
.LC12:
	.string	"3. Remove Species"
.LC13:
	.string	"4. Display Inventory"
.LC14:
	.string	"5. Exit"
.LC15:
	.string	"Enter your choice: "
	.align 8
.LC16:
	.string	"Enter species index to remove (0 to %d): "
.LC17:
	.string	"Enter the number of species: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_Evq5_envp(%rip)
	nop
.L51:
	movq	$0, _TIG_IZ_Evq5_argv(%rip)
	nop
.L52:
	movl	$0, _TIG_IZ_Evq5_argc(%rip)
	nop
	nop
.L53:
.L54:
#APP
# 223 "Mahad-Khan-14_PF-Theory-24k-0961_prob5.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Evq5--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_Evq5_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_Evq5_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_Evq5_envp(%rip)
	nop
	movq	$2, -32(%rbp)
.L92:
	cmpq	$35, -32(%rbp)
	ja	.L95
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L57(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L57(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L57:
	.long	.L75-.L57
	.long	.L74-.L57
	.long	.L73-.L57
	.long	.L95-.L57
	.long	.L72-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L71-.L57
	.long	.L95-.L57
	.long	.L70-.L57
	.long	.L69-.L57
	.long	.L68-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L67-.L57
	.long	.L95-.L57
	.long	.L66-.L57
	.long	.L95-.L57
	.long	.L65-.L57
	.long	.L95-.L57
	.long	.L64-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L63-.L57
	.long	.L62-.L57
	.long	.L61-.L57
	.long	.L60-.L57
	.long	.L59-.L57
	.long	.L95-.L57
	.long	.L95-.L57
	.long	.L58-.L57
	.long	.L95-.L57
	.long	.L56-.L57
	.text
.L66:
	movl	-72(%rbp), %eax
	cmpl	$5, %eax
	ja	.L76
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L76-.L78
	.long	.L82-.L78
	.long	.L81-.L78
	.long	.L80-.L78
	.long	.L79-.L78
	.long	.L77-.L78
	.text
.L77:
	movq	$30, -32(%rbp)
	jmp	.L83
.L79:
	movq	$12, -32(%rbp)
	jmp	.L83
.L80:
	movq	$0, -32(%rbp)
	jmp	.L83
.L81:
	movq	$27, -32(%rbp)
	jmp	.L83
.L82:
	movq	$11, -32(%rbp)
	jmp	.L83
.L76:
	movq	$16, -32(%rbp)
	nop
.L83:
	jmp	.L84
.L72:
	movl	$0, -52(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L84
.L59:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -32(%rbp)
	jmp	.L84
.L68:
	movq	-40(%rbp), %rdx
	movl	-76(%rbp), %ecx
	movq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayInventory
	movq	$26, -32(%rbp)
	jmp	.L84
.L71:
	movq	-48(%rbp), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	addl	$1, -56(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L84
.L74:
	movq	-48(%rbp), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-52(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	addl	$1, -52(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L84
.L67:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -32(%rbp)
	jmp	.L84
.L63:
	movl	-72(%rbp), %eax
	cmpl	$5, %eax
	je	.L85
	movq	$22, -32(%rbp)
	jmp	.L84
.L85:
	movq	$28, -32(%rbp)
	jmp	.L84
.L69:
	movq	-40(%rbp), %rdx
	movl	-76(%rbp), %ecx
	movq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	addSupplies
	movq	$26, -32(%rbp)
	jmp	.L84
.L62:
	movl	-76(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-40(%rbp), %rdx
	movl	-68(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-40(%rbp), %rcx
	movl	-64(%rbp), %edx
	movl	-68(%rbp), %esi
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	updateSupply
	movq	$26, -32(%rbp)
	jmp	.L84
.L64:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -32(%rbp)
	jmp	.L84
.L61:
	movl	$0, -56(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L84
.L58:
	movq	-40(%rbp), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jge	.L87
	movq	$1, -32(%rbp)
	jmp	.L84
.L87:
	movq	$8, -32(%rbp)
	jmp	.L84
.L70:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$29, -32(%rbp)
	jmp	.L84
.L75:
	movl	-76(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-60(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-60(%rbp), %ecx
	leaq	-40(%rbp), %rdx
	leaq	-76(%rbp), %rsi
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	removeSpecies
	movq	$26, -32(%rbp)
	jmp	.L84
.L56:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-76(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	-76(%rbp), %eax
	cltq
	movl	$4, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	-76(%rbp), %edx
	movq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	initializeInventory
	movq	$22, -32(%rbp)
	jmp	.L84
.L60:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L93
	jmp	.L94
.L73:
	movq	$35, -32(%rbp)
	jmp	.L84
.L65:
	movl	-76(%rbp), %eax
	cmpl	%eax, -56(%rbp)
	jge	.L90
	movq	$4, -32(%rbp)
	jmp	.L84
.L90:
	movq	$10, -32(%rbp)
	jmp	.L84
.L95:
	nop
.L84:
	jmp	.L92
.L94:
	call	__stack_chk_fail@PLT
.L93:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	initializeInventory
	.type	initializeInventory, @function
initializeInventory:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$5, -8(%rbp)
.L106:
	cmpq	$6, -8(%rbp)
	je	.L97
	cmpq	$6, -8(%rbp)
	ja	.L107
	cmpq	$5, -8(%rbp)
	je	.L99
	cmpq	$5, -8(%rbp)
	ja	.L107
	cmpq	$1, -8(%rbp)
	je	.L100
	cmpq	$2, -8(%rbp)
	je	.L108
	jmp	.L107
.L100:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L102
.L97:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L103
	movq	$1, -8(%rbp)
	jmp	.L102
.L103:
	movq	$2, -8(%rbp)
	jmp	.L102
.L99:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L102
.L107:
	nop
.L102:
	jmp	.L106
.L108:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	initializeInventory, .-initializeInventory
	.section	.rodata
.LC18:
	.string	"\n--- Inventory ---"
.LC19:
	.string	"  %s\n"
.LC20:
	.string	"Species %d:\n"
	.text
	.globl	displayInventory
	.type	displayInventory, @function
displayInventory:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$11, -8(%rbp)
.L126:
	cmpq	$13, -8(%rbp)
	ja	.L127
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L112(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L112(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L112:
	.long	.L119-.L112
	.long	.L118-.L112
	.long	.L117-.L112
	.long	.L127-.L112
	.long	.L116-.L112
	.long	.L128-.L112
	.long	.L127-.L112
	.long	.L127-.L112
	.long	.L127-.L112
	.long	.L127-.L112
	.long	.L114-.L112
	.long	.L113-.L112
	.long	.L127-.L112
	.long	.L111-.L112
	.text
.L116:
	movl	-16(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L120
	movq	$10, -8(%rbp)
	jmp	.L122
.L120:
	movq	$5, -8(%rbp)
	jmp	.L122
.L118:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L122
.L113:
	movq	$1, -8(%rbp)
	jmp	.L122
.L111:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L122
.L114:
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L122
.L119:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L124
	movq	$13, -8(%rbp)
	jmp	.L122
.L124:
	movq	$2, -8(%rbp)
	jmp	.L122
.L117:
	addl	$1, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L122
.L127:
	nop
.L122:
	jmp	.L126
.L128:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	displayInventory, .-displayInventory
	.section	.rodata
.LC21:
	.string	"Invalid supply index!"
	.align 8
.LC22:
	.string	"Enter new name for the supply: "
	.text
	.globl	updateSupply
	.type	updateSupply, @function
updateSupply:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$168, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -152(%rbp)
	movl	%esi, -156(%rbp)
	movl	%edx, -160(%rbp)
	movq	%rcx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$11, -136(%rbp)
.L151:
	cmpq	$11, -136(%rbp)
	ja	.L154
	movq	-136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L132(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L132(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L132:
	.long	.L155-.L132
	.long	.L141-.L132
	.long	.L140-.L132
	.long	.L139-.L132
	.long	.L155-.L132
	.long	.L155-.L132
	.long	.L136-.L132
	.long	.L135-.L132
	.long	.L154-.L132
	.long	.L155-.L132
	.long	.L133-.L132
	.long	.L131-.L132
	.text
.L141:
	cmpl	$0, -160(%rbp)
	jns	.L144
	movq	$3, -136(%rbp)
	jmp	.L146
.L144:
	movq	$2, -136(%rbp)
	jmp	.L146
.L139:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -136(%rbp)
	jmp	.L146
.L131:
	cmpl	$0, -156(%rbp)
	jns	.L147
	movq	$10, -136(%rbp)
	jmp	.L146
.L147:
	movq	$1, -136(%rbp)
	jmp	.L146
.L136:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-160(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-160(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	(%rdx,%rax), %rbx
	leaq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, (%rbx)
	movq	$9, -136(%rbp)
	jmp	.L146
.L133:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -136(%rbp)
	jmp	.L146
.L135:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -136(%rbp)
	jmp	.L146
.L140:
	movl	-156(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -160(%rbp)
	jl	.L149
	movq	$7, -136(%rbp)
	jmp	.L146
.L149:
	movq	$6, -136(%rbp)
	jmp	.L146
.L154:
	nop
.L146:
	jmp	.L151
.L155:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L153
	call	__stack_chk_fail@PLT
.L153:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	updateSupply, .-updateSupply
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
